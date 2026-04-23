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
  PRIMARY   (IDs 0x100, 0x200)  — '<fffBxxx'  — 16 bytes
    bytes  0– 3 : temperature_C   float32
    bytes  4– 7 : vibration_x_g   float32
    bytes  8–11 : speed_rpm / flow_rate  float32
    byte  12    : fault_id         uint8
    bytes 13–15 : padding
 
  SECONDARY (IDs 0x101, 0x201)  — '<ff'  — 8 bytes
    bytes 0–3 : ambient_temp   float32
    bytes 4–7 : humidity       float32
"""
 
from __future__ import annotations
 
import struct
import threading
import time
import sys
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Any
 
# ── Third-party — graceful import errors ──────────────────────────────────────
try:
    import can
except ImportError:
    print("[ERROR] python-can not installed.  Run:  pip install python-can")
    sys.exit(1)
 
try:
    from flask import Flask, jsonify
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
 
PRIMARY_FMT   = '<fffBxxx'   # 16 bytes: temp, vib, speed/flow, fault_id, pad
SECONDARY_FMT = '<ff'        # 8  bytes: ambient_temp, humidity
 
PRIMARY_SIZE   = struct.calcsize(PRIMARY_FMT)    # == 16
SECONDARY_SIZE = struct.calcsize(SECONDARY_FMT)  # == 8
 
FAULT_NAMES = {0: "NORMAL", 1: "BEARING_FAULT",
               2: "STATOR_FAULT", 3: "ROTOR_BAR_FAULT"}
 
HISTORY_LEN = 60  # readings kept per motor for /api/history
 
# Anomaly score thresholds (IsolationForest.decision_function values)
# decision_function > 0   → far from anomalies → NORMAL
# decision_function ∈ (-0.10, 0] → borderline → WARNING
# decision_function < -0.10 → clearly anomalous → CRITICAL
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
    "motor1"   : _motor_template(),
    "motor2"   : _motor_template(),
    "alerts"   : deque(maxlen=100),    # alert log — last 100 anomaly events
    "history1" : deque(maxlen=HISTORY_LEN),
    "history2" : deque(maxlen=HISTORY_LEN),
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
#   • Motor 1: speed ≈ 1480 RPM, temp ≈ 65 °C, vibration ≈ 0.05 g RMS
#   • Motor 2: speed translated to flow rate ≈ 100 L/min, temp ≈ 63 °C
# A shared model would learn a blended normal that incorrectly flags the pump's
# lower temperature or labels motor oscillations as anomalies.
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
# decision_function(x) = mean path length (normalised) — positive values
# indicate normal, negative indicate anomalous. contamination=0.05 sets the
# decision boundary so ~5% of training data is treated as anomalous (guards
# against outliers in the synthetic normal data).
 
class MotorAnomalyDetector:
    """Wrapper around IsolationForest for one motor."""
 
    def __init__(self, motor_id: str, contamination: float = 0.05):
        self.motor_id     = motor_id
        self.model        = IsolationForest(
            n_estimators  = 100,       # 100 trees — good bias-variance tradeoff
            contamination = contamination,
            random_state  = 42,
        )
        self.trained      = False
        self.train_count  = 0
 
    def _build_feature_vector(self, temp: float, vib: float,
                               variable: float) -> np.ndarray:
        """
        3-D feature vector: [temperature_C, vibration_x_g, speed_rpm or flow_rate].
        Kept deliberately small — IsolationForest works well in low dimensions,
        and these three channels carry the strongest fault signatures.
        """
        return np.array([[temp, vib, variable]], dtype=np.float32)
 
    def train(self, n_samples: int = 200):
        """
        Auto-train on synthetic normal data at startup.
        No separate training step — the gateway is self-configuring.
 
        WHY synthetic training?
        In a real deployment the gateway would be installed before any faults
        occur, so only normal data is available initially. Synthetic normal data
        lets us initialise the model before the first CAN frame arrives.
        In production you'd replace/augment with real baseline data after 24 h
        of confirmed normal operation.
        """
        print(f"[ML] Training IsolationForest for {self.motor_id} "
              f"on {n_samples} synthetic normal samples …")
 
        if _SIMULATOR_AVAILABLE:
            df = generate_scenario("normal", duration_s=n_samples / 1000.0,
                                   fs=1000.0, seed=42)
            # Motor 2 variance: slightly cooler, rougher vibration, has flow
            if self.motor_id == "motor2":
                temps  = df["temperature_C"].values[:n_samples] * 0.97
                vibs   = df["vibration_x_g"].values[:n_samples] * 1.05
                speeds = df["speed_rpm"].values[:n_samples] / 14.8  # → flow L/min
            else:
                temps  = df["temperature_C"].values[:n_samples]
                vibs   = df["vibration_x_g"].values[:n_samples]
                speeds = df["speed_rpm"].values[:n_samples]
        else:
            # Fallback: Gaussian around known healthy operating point
            rng = np.random.default_rng(42)
            if self.motor_id == "motor2":
                temps  = rng.normal(63.0, 1.5, n_samples)
                vibs   = rng.normal(0.053, 0.006, n_samples)
                speeds = rng.normal(100.0, 2.0, n_samples)
            else:
                temps  = rng.normal(65.0, 1.5, n_samples)
                vibs   = rng.normal(0.050, 0.006, n_samples)
                speeds = rng.normal(1480.0, 3.0, n_samples)
 
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
            > 0   → clearly normal
            0  to -0.10 → borderline (WARNING)
            < -0.10     → anomalous  (CRITICAL)
 
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
 
def _unpack_primary(data: bytes) -> tuple[float, float, float, int]:
    """Unpack PRIMARY_FMT → (temp, vib, variable, fault_id)."""
    if len(data) < PRIMARY_SIZE:
        raise ValueError(f"Primary frame too short: {len(data)} < {PRIMARY_SIZE}")
    temp, vib, variable, fault_id = struct.unpack_from(PRIMARY_FMT, data)
    return float(temp), float(vib), float(variable), int(fault_id)
 
 
def _unpack_secondary(data: bytes) -> tuple[float, float]:
    """Unpack SECONDARY_FMT → (ambient_temp, humidity)."""
    if len(data) < SECONDARY_SIZE:
        raise ValueError(f"Secondary frame too short: {len(data)} < {SECONDARY_SIZE}")
    amb, hum = struct.unpack_from(SECONDARY_FMT, data)
    return float(amb), float(hum)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CAN FD Listener (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────
 
def _handle_primary(motor_key: str, history_key: str,
                    detector: MotorAnomalyDetector,
                    temp: float, vib: float, variable: float,
                    fault_id: int, ts: str, is_motor2: bool):
    """
    Called inside the listener thread after unpacking a PRIMARY frame.
    Runs anomaly detection and updates shared state (under Lock).
    """
    score, status = detector.score(temp, vib, variable)
 
    snapshot = {
        "temperature_C" : round(temp, 2),
        "vibration_x_g" : round(vib, 4),
        "ambient_temp_C": None,     # filled when SECONDARY arrives
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
 
    # History entry (lightweight — no None fields)
    hist_entry = {
        "time"          : ts,
        "temperature_C" : snapshot["temperature_C"],
        "vibration_x_g" : snapshot["vibration_x_g"],
        "anomaly_score" : snapshot["anomaly_score"],
        "anomaly_status": status,
        **({"speed_rpm"    : snapshot["speed_rpm"]}    if not is_motor2 else {}),
        **({"flow_rate_Lm": snapshot["flow_rate_Lm"]} if is_motor2 else {}),
    }
 
    with _lock:
        prev_rx = state[motor_key].get("rx_count", 0)
        state[motor_key].update(snapshot)
        state[motor_key]["rx_count"] = prev_rx + 1
        state[history_key].append(hist_entry)
 
        # Log anomaly events to alert queue
        if status in ("WARNING", "CRITICAL"):
            alert = {
                "timestamp"    : ts,
                "motor"        : motor_key,
                "status"       : status,
                "anomaly_score": round(score, 6),
                "fault_name"   : FAULT_NAMES.get(fault_id, "UNKNOWN"),
                "temperature_C": round(temp, 2),
                "vibration_x_g": round(vib, 4),
            }
            state["alerts"].append(alert)
 
    # Console output — written outside lock to minimise hold time
    icon = {"NORMAL": "✓", "WARNING": "⚠", "CRITICAL": "✗"}.get(status, "?")
    motor_label = motor_key.upper()
    print(f"  [{ts}] RX  {motor_label}  "
          f"temp={temp:.1f}°C  vib={vib:.4f}g  "
          f"{'rpm' if not is_motor2 else 'flow'}={variable:.1f}  "
          f"fault={FAULT_NAMES.get(fault_id,'?'):<16}  "
          f"score={score:+.4f}  [{icon} {status}]")
 
 
def can_listener(channel: str = "vcan0"):
    """
    Background daemon thread.
    Opens the python-can virtual bus in receive-only mode and processes
    all frames forever, updating shared `state` on each primary frame.
 
    In the real Raspberry Pi 4 deployment this would use:
        interface='socketcan', channel='can0'   (MCP2518FD HAT via SPI)
    The virtual bus lets us develop and test without hardware.
    """
    print("[CAN] Listener thread starting …")
 
    try:
        bus = can.Bus(
            interface = 'udp_multicast',
            channel   = '239.0.0.1',
            fd        = True,
        )
    except Exception as exc:
        print(f"[CAN] FATAL — cannot open bus: {exc}")
        return
 
    print(f"[CAN] Listening on virtual:{channel}  (Ctrl-C to stop)\n")
    print(f"  {'Timestamp':<12} Dir   Motor    Readings"
          f"                                    Score      Status")
    print(f"  {'─'*100}")
 
    for msg in bus:
        arb = msg.arbitration_id
        ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
 
        try:
            if arb == FRAME_MOTOR1_PRIMARY:
                temp, vib, speed, fault_id = _unpack_primary(msg.data)
                _handle_primary("motor1", "history1", detector1,
                                 temp, vib, speed, fault_id, ts,
                                 is_motor2=False)
 
            elif arb == FRAME_MOTOR1_SECONDARY:
                amb, hum = _unpack_secondary(msg.data)
                with _lock:
                    state["motor1"]["ambient_temp_C"] = round(amb, 2)
                    state["motor1"]["humidity_pct"]   = round(hum, 2)
 
            elif arb == FRAME_MOTOR2_PRIMARY:
                temp, vib, flow, fault_id = _unpack_primary(msg.data)
                _handle_primary("motor2", "history2", detector2,
                                 temp, vib, flow, fault_id, ts,
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
    Used by the dashboard overview panel.
 
    Example response — see sample_responses.json
    """
    with _lock:
        m1 = dict(state["motor1"])
        m2 = dict(state["motor2"])
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
    Full detail for Motor 1 (induction motor):
    temp, vib, RPM, humidity, anomaly score, status.
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
    Full detail for Motor 2 (centrifugal pump):
    temp, vib, flow_rate, humidity, anomaly score, status.
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
    Returns the last 60 readings per motor as arrays suitable for
    Chart.js time-series charts on the dashboard.
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
        "alerts"   : list(reversed(alerts)),   # newest first
    })
 
# -----------------------------------------------------------------------------
# flask_cors
# -----------------------------------------------------------------------------
from flask_cors import CORS
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    print(f"\n{'='*68}")
    print(f"  ml_gateway.py — Raspberry Pi 4 ML Gateway")
    print(f"  Isolation Forest anomaly detection per motor")
    print(f"  CAN FD → ML → Flask REST API")
    print(f"{'='*68}\n")
 
    # ── Step 1: Train ML models ───────────────────────────────────────────────
    detector1.train(n_samples=200)
    detector2.train(n_samples=200)
    print()
 
    # ── Step 2: Start CAN listener in background thread ───────────────────────
    listener_thread = threading.Thread(
        target   = can_listener,
        kwargs   = {"channel": "239.0.0.1"},
        daemon   = True,    # dies automatically when main thread exits (Ctrl-C)
        name     = "CANListener",
    )
    listener_thread.start()
    print(f"\n[MAIN] CAN listener thread started (daemon={listener_thread.daemon})")
 
    # Short grace period so listener prints its banner before Flask banner
    time.sleep(0.3)
 
    # ── Step 3: Flask on main thread ──────────────────────────────────────────
    print(f"\n[MAIN] Starting Flask REST API on http://0.0.0.0:5000\n")
    print(f"  Endpoints:")
    print(f"    GET /api/status        — both motors overview")
    print(f"    GET /api/motor/1       — Motor 1 detail")
    print(f"    GET /api/motor/2       — Motor 2 detail")
    print(f"    GET /api/history       — last {HISTORY_LEN} readings per motor")
    print(f"    GET /api/alerts        — anomaly event log")
    print()
 
    # use_reloader=False is CRITICAL — the reloader spawns a second process
    # which would open a second CAN bus connection and break the listener.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
 
 
if __name__ == "__main__":
    main()