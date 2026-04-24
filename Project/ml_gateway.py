"""
ml_gateway.py — Raspberry Pi 4 ML Gateway
==========================================
Receives CAN FD frames from the python-can virtual bus, unpacks sensor
values, runs Isolation Forest anomaly detection per machine, and exposes
results via a Flask REST API for the dashboard to poll.

Real-hardware mapping
---------------------
  CAN FD transmitter  →  ESP32 WROOM-32 nodes (one per motor)
  python-can virtual  →  MCP2518FD CAN FD HAT on Raspberry Pi 4
  Flask REST API      →  Served on RPi LAN (e.g. http://192.168.1.50:5000)
  Dashboard           →  Any browser on the same LAN

Threading model
---------------
  Thread-1 (daemon)   : CAN FD listener — receives frames, unpacks struct bytes,
                        updates shared state dict under Lock
  Thread-2 (main)     : Flask — reads shared state under Lock to serve JSON

  WHY threading.Lock?
  -------------------
  Python's GIL prevents true parallel execution of Python bytecode, but it
  does NOT protect multi-step read-modify-write operations on shared objects.
  Without a Lock, the Flask thread could read `state` mid-update (e.g. after
  Motor1 primary is written but before secondary is written) and return a
  half-updated snapshot. Lock.acquire() before any state read/write ensures
  atomic, consistent snapshots.

Frame format (mirrors can_node.py exactly)
------------------------------------------
  PRIMARY   (IDs 0x100, 0x200)  — '<ffffffBxxxxxxx'  — 32 bytes
    bytes  0– 3 : temperature_C        float32
    bytes  4– 7 : vibration_x_g        float32
    bytes  8–11 : speed_rpm / flow_Lm  float32
    bytes 12–15 : current_a_A          float32  (Phase-A stator current)
    bytes 16–19 : current_b_A          float32  (Phase-B stator current)
    bytes 20–23 : current_c_A          float32  (Phase-C stator current)
    byte  24    : fault_id             uint8
    bytes 25–31 : padding

  SECONDARY (IDs 0x101, 0x201)  — '<ff'  — 8 bytes
    bytes 0–3 : ambient_temp   float32
    bytes 4–7 : humidity       float32
"""

from __future__ import annotations

import os
import struct
import threading
import time
import sys
import smtplib
import email.message
from collections import deque
from datetime import datetime
from typing import Dict, Any

# ── Third-party — graceful import errors ──────────────────────────────────────
try:
    import can
except ImportError:
    print("[ERROR] python-can not installed.  Run:  pip install python-can")
    sys.exit(1)

try:
    from flask import Flask, jsonify, request
except ImportError:
    print("[ERROR] Flask not installed.  Run:  pip install flask")
    sys.exit(1)

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
except ImportError:
    print("[ERROR] scikit-learn / numpy not installed.")
    print("        Run:  pip install scikit-learn numpy")
    sys.exit(1)

try:
    from flask_cors import CORS
except ImportError:
    print("[ERROR] flask-cors not installed.  Run:  pip install flask-cors")
    sys.exit(1)

# sensor_simulator is needed only for training data — import is optional
try:
    from sensor_simulator import generate_scenario
    _SIMULATOR_AVAILABLE = True
except ImportError:
    _SIMULATOR_AVAILABLE = False
    print("[WARN] sensor_simulator.py not found — will use synthetic Gaussian "
          "training data instead of realistic motor data.")


# ─────────────────────────────────────────────────────────────────────────────
# Constants — must mirror can_node.py exactly
# ─────────────────────────────────────────────────────────────────────────────

FRAME_MOTOR1_PRIMARY   = 0x100
FRAME_MOTOR1_SECONDARY = 0x101
FRAME_MOTOR2_PRIMARY   = 0x200
FRAME_MOTOR2_SECONDARY = 0x201

# CRITICAL: this string must be character-for-character identical to can_node.py
PRIMARY_FMT   = '<ffffffBxxxxxxx'  # 32 bytes: 6xfloat32 + uint8 + 7 pad
SECONDARY_FMT = '<ff'              # 8  bytes: 2xfloat32

PRIMARY_SIZE   = struct.calcsize(PRIMARY_FMT)    # == 32
SECONDARY_SIZE = struct.calcsize(SECONDARY_FMT)  # == 8

FAULT_NAMES = {
    0: "NORMAL",
    1: "BEARING_FAULT",
    2: "STATOR_FAULT",
    3: "ROTOR_BAR_FAULT",
}

HISTORY_LEN = 60   # readings kept per motor for /api/history

# Anomaly score thresholds (IsolationForest.decision_function values)
# decision_function > 0.0   → far from anomaly boundary → NORMAL
# decision_function in (-0.10, 0.0] → borderline         → WARNING
# decision_function < -0.10          → clearly anomalous  → CRITICAL
THRESHOLD_WARNING  = 0.0
THRESHOLD_CRITICAL = -0.10


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────
#
# `state` holds the latest snapshot for each motor plus a history deque.
# It is the ONLY shared mutable object between the listener thread and Flask.

_lock: threading.Lock = threading.Lock()


def _motor_template() -> Dict[str, Any]:
    return {
        "temperature_C"  : None,
        "vibration_x_g"  : None,
        "current_a_A"    : None,   # Phase-A stator current (A)
        "current_b_A"    : None,   # Phase-B stator current (A)
        "current_c_A"    : None,   # Phase-C stator current (A)
        "speed_rpm"      : None,   # Motor 1 only
        "flow_rate_Lm"   : None,   # Motor 2 only
        "ambient_temp_C" : None,
        "humidity_pct"   : None,
        "fault_id"       : None,
        "fault_name"     : None,
        "anomaly_score"  : None,   # raw decision_function value
        "anomaly_status" : "INITIALISING",
        "last_updated"   : None,
        "rx_count"       : 0,
    }


state: Dict[str, Any] = {
    "motor1"        : _motor_template(),
    "motor2"        : _motor_template(),
    "alerts"        : deque(maxlen=100),       # last 100 anomaly events
    "history1"      : deque(maxlen=HISTORY_LEN),
    "history2"      : deque(maxlen=HISTORY_LEN),
    "gateway_start" : datetime.now().isoformat(timespec="seconds"),
}


# ─────────────────────────────────────────────────────────────────────────────
# ML Models — one Isolation Forest per motor
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY a separate model per motor?
# --------------------------------
# Motor 1 (induction motor) and Motor 2 (centrifugal pump) have fundamentally
# different normal operating envelopes:
#   Motor 1: speed ~1480 RPM, temp ~65 C, vibration ~0.05 g RMS
#   Motor 2: speed translated to flow rate ~100 L/min, temp ~63 C
# A shared model would learn a blended normal that incorrectly flags one motor.
# Separate models means each machine's "normal" is learned independently,
# giving higher specificity and lower false-positive rate.
#
# HOW Isolation Forest works (unsupervised — no labels needed)
# ------------------------------------------------------------
# The algorithm builds an ensemble of random binary trees. Each tree:
#   1. Selects a random feature.
#   2. Selects a random split value between feature min and max.
#   3. Recursively partitions the data until each point is isolated.
# ANOMALIES are isolated in fewer splits (shorter path length) because they
# live in sparse regions. NORMAL points require many splits.
# decision_function(x) = mean path length (normalised) — positive = normal,
# negative = anomalous. contamination=0.05 sets the decision boundary so
# ~5% of training data is treated as anomalous.

class MotorAnomalyDetector:
    """Wrapper around IsolationForest for one motor."""

    def __init__(self, motor_id: str, contamination: float = 0.05):
        self.motor_id    = motor_id
        self.model       = IsolationForest(
            n_estimators  = 100,
            contamination = contamination,
            random_state  = 42,
        )
        self.trained     = False
        self.train_count = 0

    def _build_feature_vector(self, temp: float, vib: float,
                               variable: float) -> np.ndarray:
        """
        3-D feature vector: [temperature_C, vibration_x_g, speed_rpm or flow_rate].
        Kept deliberately small — IsolationForest works well in low dimensions,
        and these three channels carry the strongest fault signatures.

        NOTE: current_a/b/c_A are stored and exposed via API for dashboard
        visualisation but are NOT used as ML features here. Adding current
        to the feature vector is a future improvement (requires retraining).
        """
        return np.array([[temp, vib, variable]], dtype=np.float32)

    def train(self, n_samples: int = 200):
        """
        Auto-train on synthetic normal data at startup.
        No separate training step — the gateway is self-configuring.

        WHY synthetic training?
        In a real deployment the gateway is installed before any faults occur,
        so only normal data is available initially. Synthetic normal data lets
        us initialise the model before the first CAN frame arrives. In
        production you would replace/augment with real baseline data after
        24 h of confirmed normal operation.
        """
        print(f"[ML] Training IsolationForest for {self.motor_id} "
              f"on {n_samples} synthetic normal samples ...")

        if _SIMULATOR_AVAILABLE:
            df = generate_scenario("normal", duration_s=n_samples / 1000.0,
                                   fs=1000.0, seed=42)
            if self.motor_id == "motor2":
                temps  = df["temperature_C"].values[:n_samples] * 0.97
                vibs   = df["vibration_x_g"].values[:n_samples] * 1.05
                speeds = df["speed_rpm"].values[:n_samples] / 14.8
            else:
                temps  = df["temperature_C"].values[:n_samples]
                vibs   = df["vibration_x_g"].values[:n_samples]
                speeds = df["speed_rpm"].values[:n_samples]
        else:
            rng = np.random.default_rng(42)
            if self.motor_id == "motor2":
                temps  = rng.normal(63.0,   1.5,  n_samples)
                vibs   = rng.normal(0.053,  0.006, n_samples)
                speeds = rng.normal(100.0,  2.0,  n_samples)
            else:
                temps  = rng.normal(65.0,   1.5,  n_samples)
                vibs   = rng.normal(0.050,  0.006, n_samples)
                speeds = rng.normal(1480.0, 3.0,  n_samples)

        X = np.column_stack([temps, vibs, speeds])
        self.model.fit(X)
        self.trained     = True
        self.train_count = n_samples
        print(f"[ML] {self.motor_id} model trained. "
              f"decision_function range on training data: "
              f"[{self.model.decision_function(X).min():.4f}, "
              f"{self.model.decision_function(X).max():.4f}]")

    def score(self, temp: float, vib: float, variable: float) -> tuple[float, str]:
        """
        Returns (score, status_label).

        score = decision_function value (float):
            > 0.0        → clearly normal
            -0.10 to 0.0 → borderline (WARNING)
            < -0.10      → anomalous  (CRITICAL)

        The decision_function measures the normalised average path length
        across all trees. Unlike predict() which returns +1/-1, this gives
        a continuous confidence measure useful for dashboard gauges and
        research paper ROC curves.
        """
        if not self.trained:
            return (0.0, "INITIALISING")

        X     = self._build_feature_vector(temp, vib, variable)
        score = float(self.model.decision_function(X)[0])

        if score > THRESHOLD_WARNING:
            status = "NORMAL"
        elif score > THRESHOLD_CRITICAL:
            status = "WARNING"
        else:
            status = "CRITICAL"

        return (score, status)


# Instantiate one detector per motor at module level
detector1 = MotorAnomalyDetector("motor1")
detector2 = MotorAnomalyDetector("motor2")


# ─────────────────────────────────────────────────────────────────────────────
# Frame unpacking
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_primary(data: bytes) -> tuple[float, float, float, float, float, float, int]:
    """
    Unpack PRIMARY_FMT bytes into sensor values.
    Returns (temp, vib, variable, current_a, current_b, current_c, fault_id).
    """
    if len(data) < PRIMARY_SIZE:
        raise ValueError(f"Primary frame too short: {len(data)} < {PRIMARY_SIZE}")
    temp, vib, variable, ia, ib, ic, fault_id = struct.unpack_from(PRIMARY_FMT, data)
    return float(temp), float(vib), float(variable), float(ia), float(ib), float(ic), int(fault_id)


def _unpack_secondary(data: bytes) -> tuple[float, float]:
    """Unpack SECONDARY_FMT → (ambient_temp, humidity)."""
    if len(data) < SECONDARY_SIZE:
        raise ValueError(f"Secondary frame too short: {len(data)} < {SECONDARY_SIZE}")
    amb, hum = struct.unpack_from(SECONDARY_FMT, data)
    return float(amb), float(hum)


# ─────────────────────────────────────────────────────────────────────────────
# Primary frame handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_primary(motor_key: str, history_key: str,
                    detector: MotorAnomalyDetector,
                    temp: float, vib: float, variable: float,
                    ia: float, ib: float, ic: float,
                    fault_id: int, ts: str, is_motor2: bool):
    """
    Called inside the listener thread after unpacking a PRIMARY frame.
    Runs anomaly detection and updates shared state under Lock.

    ia, ib, ic — three-phase stator currents (A).
    Stored in state snapshot, hist_entry, and alert log so the dashboard
    Sensor Trends tab can display phase-by-phase current charts.
    Fault signatures visible in current:
      stator_fault    → phase-A imbalance, 3rd/5th harmonic pollution
      rotor_bar_fault → sidebands at (1+-2ks)*50 Hz in all three phases
      bearing_fault   → no direct current effect (useful negative case)
    """
    score, status = detector.score(temp, vib, variable)

    snapshot = {
        "temperature_C" : round(temp, 2),
        "vibration_x_g" : round(vib, 4),
        "current_a_A"   : round(ia, 4),
        "current_b_A"   : round(ib, 4),
        "current_c_A"   : round(ic, 4),
        "ambient_temp_C": None,    # filled when SECONDARY frame arrives
        "humidity_pct"  : None,
        "fault_id"      : fault_id,
        "fault_name"    : FAULT_NAMES.get(fault_id, "UNKNOWN"),
        "anomaly_score" : round(score, 6),
        "anomaly_status": status,
        "last_updated"  : ts,
    }
    if is_motor2:
        snapshot["flow_rate_Lm"] = round(variable, 2)
        snapshot["speed_rpm"]    = None
    else:
        snapshot["speed_rpm"]    = round(variable, 1)
        snapshot["flow_rate_Lm"] = None

    # History entry — all fields the dashboard needs for trend charts
    hist_entry = {
        "time"          : ts,
        "temperature_C" : snapshot["temperature_C"],
        "vibration_x_g" : snapshot["vibration_x_g"],
        "current_a_A"   : snapshot["current_a_A"],
        "current_b_A"   : snapshot["current_b_A"],
        "current_c_A"   : snapshot["current_c_A"],
        "anomaly_score" : snapshot["anomaly_score"],
        "anomaly_status": status,
        **({"speed_rpm"    : snapshot["speed_rpm"]}    if not is_motor2 else {}),
        **({"flow_rate_Lm": snapshot["flow_rate_Lm"]} if is_motor2     else {}),
    }

    with _lock:
        prev_rx = state[motor_key].get("rx_count", 0)
        state[motor_key].update(snapshot)
        state[motor_key]["rx_count"] = prev_rx + 1
        state[history_key].append(hist_entry)

        # Log WARNING and CRITICAL events to alert queue
        if status in ("WARNING", "CRITICAL"):
            alert = {
                "timestamp"    : ts,
                "motor"        : motor_key,
                "status"       : status,
                "anomaly_score": round(score, 6),
                "fault_name"   : FAULT_NAMES.get(fault_id, "UNKNOWN"),
                "temperature_C": round(temp, 2),
                "vibration_x_g": round(vib, 4),
                "current_a_A"  : round(ia, 4),
            }
            state["alerts"].append(alert)

    # Console output — written outside lock to minimise hold time
    icon        = {"NORMAL": "✓", "WARNING": "⚠", "CRITICAL": "✗"}.get(status, "?")
    motor_label = motor_key.upper()
    print(f"  [{ts}] RX  {motor_label}  "
          f"temp={temp:.1f}C  vib={vib:.4f}g  "
          f"{'rpm' if not is_motor2 else 'flow'}={variable:.1f}  "
          f"Ia={ia:.2f}A  "
          f"fault={FAULT_NAMES.get(fault_id, '?'):<16}  "
          f"score={score:+.4f}  [{icon} {status}]")


# ─────────────────────────────────────────────────────────────────────────────
# CAN FD Listener (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

def can_listener(channel: str = "vcan0"):
    """
    Background daemon thread.
    Opens the python-can virtual bus in receive-only mode and processes
    all frames forever, updating shared `state` on each primary frame.

    In the real Raspberry Pi 4 deployment this would use:
        interface='socketcan', channel='can0'   (MCP2518FD HAT via SPI)
    The virtual bus lets us develop and test without hardware.
    """
    print("[CAN] Listener thread starting ...")

    try:
        bus = can.Bus(
            interface = 'udp_multicast',
            channel   = '239.0.0.1',
            fd        = True,
        )
    except Exception as exc:
        print(f"[CAN] FATAL — cannot open bus: {exc}")
        return

    print(f"[CAN] Listening on udp_multicast:239.0.0.1  (Ctrl-C to stop)\n")
    print(f"  {'Timestamp':<12} Dir   Motor    Readings"
          f"                                         Score      Status")
    print(f"  {'─'*105}")

    for msg in bus:
        arb = msg.arbitration_id
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        try:
            if arb == FRAME_MOTOR1_PRIMARY:
                temp, vib, speed, ia, ib, ic, fault_id = _unpack_primary(msg.data)
                _handle_primary("motor1", "history1", detector1,
                                temp, vib, speed, ia, ib, ic, fault_id, ts,
                                is_motor2=False)

            elif arb == FRAME_MOTOR1_SECONDARY:
                amb, hum = _unpack_secondary(msg.data)
                with _lock:
                    state["motor1"]["ambient_temp_C"] = round(amb, 2)
                    state["motor1"]["humidity_pct"]   = round(hum, 2)

            elif arb == FRAME_MOTOR2_PRIMARY:
                temp, vib, flow, ia, ib, ic, fault_id = _unpack_primary(msg.data)
                _handle_primary("motor2", "history2", detector2,
                                temp, vib, flow, ia, ib, ic, fault_id, ts,
                                is_motor2=True)

            elif arb == FRAME_MOTOR2_SECONDARY:
                amb, hum = _unpack_secondary(msg.data)
                with _lock:
                    state["motor2"]["ambient_temp_C"] = round(amb, 2)
                    state["motor2"]["humidity_pct"]   = round(hum, 2)

        except Exception as exc:
            print(f"  [{ts}] [WARN] Frame decode error (ID=0x{arb:03X}): {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Flask REST API
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


def _motor_snapshot(motor_key: str) -> Dict[str, Any]:
    """Thread-safe read of latest motor state."""
    with _lock:
        return dict(state[motor_key])


def _history_snapshot(history_key: str) -> list:
    with _lock:
        return list(state[history_key])


def _alerts_snapshot() -> list:
    with _lock:
        return list(state["alerts"])


@app.route("/api/status", methods=["GET"])
def api_status():
    """
    GET /api/status
    Returns the latest readings + anomaly status for BOTH motors.
    Includes current_a_A, current_b_A, current_c_A per motor.
    Used by the dashboard overview panel.
    """
    with _lock:
        m1    = dict(state["motor1"])
        m2    = dict(state["motor2"])
        start = state["gateway_start"]

    return jsonify({
        "gateway_start": start,
        "timestamp"    : datetime.now().isoformat(timespec="milliseconds"),
        "motor1"       : m1,
        "motor2"       : m2,
    })


@app.route("/api/motor/1", methods=["GET"])
def api_motor1():
    """
    GET /api/motor/1
    Full detail for Motor 1 (induction motor).
    """
    return jsonify({
        "motor_id" : 1,
        "type"     : "Three-Phase Induction Motor",
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        **_motor_snapshot("motor1"),
    })


@app.route("/api/motor/2", methods=["GET"])
def api_motor2():
    """
    GET /api/motor/2
    Full detail for Motor 2 (centrifugal pump).
    """
    return jsonify({
        "motor_id" : 2,
        "type"     : "Centrifugal Pump Motor",
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        **_motor_snapshot("motor2"),
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    GET /api/history
    Returns the last 60 readings per motor for Chart.js trend charts.
    Each entry includes: time, temperature_C, vibration_x_g,
    current_a_A, current_b_A, current_c_A, anomaly_score,
    anomaly_status, and speed_rpm (motor1) or flow_rate_Lm (motor2).
    """
    return jsonify({
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "motor1"   : _history_snapshot("history1"),
        "motor2"   : _history_snapshot("history2"),
    })


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    """
    GET /api/alerts
    List of recent anomaly events (WARNING + CRITICAL), newest first.
    Used by the dashboard alert log panel.
    """
    alerts = _alerts_snapshot()
    return jsonify({
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "count"    : len(alerts),
        "alerts"   : list(reversed(alerts)),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Email alert endpoint
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY smtplib over a third-party library?
# smtplib is part of Python's standard library — no extra pip install,
# works on a fresh Raspberry Pi OS image without internet access.
#
# Configure via environment variables before launching ml_gateway.py:
#
#   export SMTP_HOST=smtp.gmail.com
#   export SMTP_PORT=587
#   export SMTP_USER=your@gmail.com
#   export SMTP_PASS=your_app_password    (Gmail App Password, not account pwd)
#   export SMTP_FROM=your@gmail.com
#
# For Gmail: enable 2FA then create an App Password at:
#   https://myaccount.google.com/apppasswords


def _send_smtp(recipient: str, alert: dict) -> tuple[bool, str]:
    """
    Send one SMTP email for the given alert dict.
    Returns (success: bool, message: str).
    """
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "umesh.marvellous@gmail.com")
    smtp_pass = os.environ.get("SMTP_PASS", "wmaj ehoa fvbr magx")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)

    if not smtp_host or not smtp_user or not smtp_pass:
        return False, ("SMTP not configured. "
                       "Set SMTP_HOST, SMTP_USER, SMTP_PASS environment variables.")

    motor  = (alert.get("motor", "") or "").replace("motor", "Motor ")
    fault  = (alert.get("fault_name", "UNKNOWN") or "").replace("_", " ")
    status = alert.get("status", "UNKNOWN")
    score  = alert.get("anomaly_score", "-")
    temp   = alert.get("temperature_C", "-")
    vib    = alert.get("vibration_x_g", "-")
    ia     = alert.get("current_a_A", "-")
    ts     = alert.get("timestamp", "-")

    subject = f"[{status}] Motor Fault Alert — {motor}: {fault}"
    body = (
        f"Smart Industrial Automation — Fault Alert\n"
        f"{'=' * 52}\n\n"
        f"Status         : {status}\n"
        f"Motor          : {motor}\n"
        f"Fault          : {fault}\n"
        f"Timestamp      : {ts}\n\n"
        f"Anomaly Score  : {score}\n"
        f"Temperature    : {temp} C\n"
        f"Vibration X    : {vib} g\n"
        f"Current Ph-A   : {ia} A\n\n"
        f"{'─' * 52}\n"
        f"Generated by ml_gateway.py — Isolation Forest anomaly detection\n"
        f"Dashboard      : http://localhost:5000\n"
    )

    msg = email.message.EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = smtp_from
    msg["To"]      = recipient
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, f"Email delivered to {recipient}"
    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed. Check SMTP_USER / SMTP_PASS."
    except Exception as exc:
        return False, f"SMTP error: {exc}"


@app.route("/api/email_alert", methods=["POST"])
def api_email_alert():
    """
    POST /api/email_alert
    Called automatically by dashboard.html when a WARNING/CRITICAL alert fires.

    Request body (JSON):
    {
      "recipient"    : "engineer@plant.com",
      "motor"        : "motor1",
      "status"       : "CRITICAL",
      "fault_name"   : "BEARING_FAULT",
      "anomaly_score": -0.142,
      "temperature_C": 87.3,
      "vibration_x_g": 0.412,
      "current_a_A"  : 41.2,
      "timestamp"    : "14:22:07.345"
    }

    Returns 200 on success, 503 if SMTP not configured, 400 on bad request.
    """
    try:
        data      = request.get_json(force=True) or {}
        recipient = (data.get("recipient") or "").strip()
        if not recipient or "@" not in recipient:
            return jsonify({"ok": False,
                            "message": "Missing or invalid recipient email."}), 400

        success, message = _send_smtp(recipient, data)
        if success:
            print(f"[EMAIL] Sent to {recipient}: "
                  f"{data.get('status')} / {data.get('fault_name')}")
            return jsonify({"ok": True, "message": message}), 200
        else:
            print(f"[EMAIL] Failed: {message}")
            return jsonify({"ok": False, "message": message}), 503

    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*68}")
    print(f"  ml_gateway.py — Raspberry Pi 4 ML Gateway")
    print(f"  Isolation Forest anomaly detection per motor")
    print(f"  CAN FD → ML → Flask REST API")
    print(f"  PRIMARY frame : '<ffffffBxxxxxxx'  32 bytes")
    print(f"    Fields      : temp, vib, spd/flow, Ia, Ib, Ic, fault_id")
    print(f"{'='*68}\n")

    # ── Step 1: Train ML models ───────────────────────────────────────────────
    detector1.train(n_samples=200)
    detector2.train(n_samples=200)
    print()

    # ── Step 2: Start CAN listener in background thread ───────────────────────
    listener_thread = threading.Thread(
        target = can_listener,
        kwargs = {"channel": "239.0.0.1"},
        daemon = True,   # dies automatically when main thread exits (Ctrl-C)
        name   = "CANListener",
    )
    listener_thread.start()
    print(f"\n[MAIN] CAN listener thread started (daemon={listener_thread.daemon})")

    # Short grace period so listener prints its banner before Flask banner
    time.sleep(0.3)

    # ── Step 3: Flask on main thread ──────────────────────────────────────────
    print(f"\n[MAIN] Starting Flask REST API on http://0.0.0.0:5000\n")
    print(f"  Endpoints:")
    print(f"    GET  /api/status        — both motors overview (incl. Ia, Ib, Ic)")
    print(f"    GET  /api/motor/1       — Motor 1 full detail")
    print(f"    GET  /api/motor/2       — Motor 2 full detail")
    print(f"    GET  /api/history       — last {HISTORY_LEN} readings per motor (incl. 3-phase current)")
    print(f"    GET  /api/alerts        — anomaly event log")
    print(f"    POST /api/email_alert   — send SMTP alert email")
    print(f"\n  Email SMTP config via env vars:")
    print(f"    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM")
    print()

    # use_reloader=False is CRITICAL — the reloader spawns a second process
    # which would open a second CAN bus connection and break the listener.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()