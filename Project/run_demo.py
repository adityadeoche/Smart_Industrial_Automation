"""
run_demo.py — Smart Industrial Automation — Integration & Demo Script
=====================================================================
Single-terminal orchestrator that:
  1. Trains both Isolation Forest ML models
  2. Starts the Flask API server in a daemon thread
  3. Cycles through all 4 fault scenarios automatically
  4. Prints live console output per frame
  5. Displays a final summary table
  6. Keeps Flask alive for examiner inspection (Ctrl-C to stop)

WHY direct state update instead of virtual CAN bus?
----------------------------------------------------
The udp_multicast interface requires OS-level multicast routing.  On many
development machines (Windows, restricted Linux, WSL) multicast sockets are
blocked or unavailable.  Calling _handle_primary() directly bypasses the
network layer entirely — the demo works on any machine that can run Python.
This is architecturally equivalent: the same ML scoring + state update path
is exercised; only the frame-transport layer is replaced by a function call.

WHY Flask in a daemon thread?
------------------------------
daemon=True means the thread is automatically killed when the main thread
exits.  No orphan Flask process is left running after Ctrl-C.  The main
thread remains free to run the scenario sequencer and print console output.

Threading model
---------------
  Main thread   : demo controller — scenario sequencer, console output,
                  summary table, then blocks on input() for examiner time
  Thread-2 (daemon): Flask REST API — responds to /api/* from dashboard
  (No separate CAN listener thread — state is updated directly)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from typing import Dict, List

# ── Project imports ──────────────────────────────────────────────────────────
try:
    from sensor_simulator import generate_scenario, SCENARIOS
except ImportError:
    print("[ERROR] sensor_simulator.py not found in current directory.")
    sys.exit(1)

try:
    from ml_gateway import (
        app,
        detector1, detector2,
        state, _lock,
        _handle_primary,
        FAULT_NAMES,
    )
except ImportError as exc:
    print(f"[ERROR] Could not import ml_gateway: {exc}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Fault ID mapping  (mirrors can_node.py FAULT_ID_MAP)
# ─────────────────────────────────────────────────────────────────────────────

FAULT_ID_MAP: Dict[str, int] = {
    "normal"        : 0,
    "bearing_fault" : 1,
    "stator_fault"  : 2,
    "rotor_bar_fault": 3,
}

# Human-readable injection description per scenario (for banner)
SCENARIO_INFO: Dict[str, str] = {
    "normal"         : "Healthy baseline — all sensors in nominal range",
    "bearing_fault"  : "vib↑  temp↑  speed↓  (outer-race impact impulses)",
    "stator_fault"   : "temp↑↑  vib↑  speed↓  (I²R hotspot in windings)",
    "rotor_bar_fault": "speed↓  vib↑  temp↑   (torque pulsation)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: update secondary state (ambient temp + humidity) directly
# ─────────────────────────────────────────────────────────────────────────────

def _handle_secondary_state(motor_key: str, amb: float, hum: float) -> None:
    """
    Write ambient temperature and humidity directly to shared state.
    Replaces the SECONDARY CAN frame path for demo portability.
    Uses the same Lock as _handle_primary for thread safety.
    """
    with _lock:
        state[motor_key]["ambient_temp_C"] = round(amb, 2)
        state[motor_key]["humidity_pct"]   = round(hum, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario banner printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner(index: int, total: int, scenario: str,
                  dwell: float, interval: float) -> None:
    name = scenario.replace("_", " ").upper()
    info = SCENARIO_INFO.get(scenario, "")
    frames_est = int(dwell / interval)
    print()
    print("  ══════════════════════════════════════════════════════")
    print(f"  SCENARIO {index}/{total} — {name}")
    print(f"  Injecting : {info}")
    print(f"  Duration  : {dwell:.0f}s  |  Interval: {interval}s  "
          f"|  ~{frames_est} frames")
    print("  ══════════════════════════════════════════════════════")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame console line
# ─────────────────────────────────────────────────────────────────────────────

def _print_frame_line(ts: str, motor_label: str,
                      temp: float, vib: float, variable: float,
                      score: float, status: str, fault_name: str,
                      is_motor2: bool) -> None:
    icon = {"NORMAL": "✓", "WARNING": "⚠", "CRITICAL": "✗"}.get(status, "?")
    var_label = "flow" if is_motor2 else "rpm "
    var_unit  = "L/m" if is_motor2 else "   "
    # Score sign formatting
    score_str = f"{score:+.4f}"
    print(
        f"  [{ts}] {motor_label} | "
        f"temp={temp:6.2f}°C  vib={vib:.4f}g  "
        f"{var_label}={variable:7.1f}{var_unit}  "
        f"score={score_str}  [{icon} {status:<8}]  "
        f"fault={fault_name}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(stats: List[Dict]) -> None:
    """Print formatted summary table after all scenarios complete."""
    import urllib.request, json as _json
    # Try to get live alert count from API
    alert_count = 0
    try:
        with urllib.request.urlopen("http://localhost:5000/api/alerts",
                                    timeout=2) as r:
            data = _json.loads(r.read())
            alert_count = data.get("count", 0)
    except Exception:
        # Fall back to counting from state
        with _lock:
            alert_count = len(state["alerts"])

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║              DEMO COMPLETE — SYSTEM SUMMARY                  ║")
    print("  ╠══════════════╦══════════════╦═══════════╦════════════════════╣")
    print("  ║ Scenario     ║ Frames Sent  ║ Avg Score ║ Peak Status        ║")
    print("  ╠══════════════╬══════════════╬═══════════╬════════════════════╣")
    for s in stats:
        avg  = s["avg_score"]
        sign = "+" if avg >= 0 else "−"
        avg_str = f"{sign}{abs(avg):.4f}"
        print(
            f"  ║ {s['scenario']:<12} ║ {s['frames']:>12} ║ {avg_str:>9} "
            f"║ {s['peak_status']:<18} ║"
        )
    print("  ╚══════════════╩══════════════╩═══════════╩════════════════════╝")
    print(f"  Total alerts logged: {alert_count}")
    print()
    print("  API still live at http://localhost:5000 — press Ctrl-C to stop")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Flask startup helper
# ─────────────────────────────────────────────────────────────────────────────

def _start_flask() -> threading.Thread:
    """
    Launch Flask in a daemon thread.

    WHY daemon=True?
    ----------------
    A daemon thread is automatically killed when the main thread exits.
    This means pressing Ctrl-C (which raises KeyboardInterrupt in main)
    will cleanly terminate Flask without any orphan server process.
    Without daemon=True, the process would hang even after Ctrl-C because
    Flask's internal WSGI server keeps the non-daemon thread alive.
    """
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host        = "0.0.0.0",
            port        = 5000,
            debug       = False,
            use_reloader= False,   # CRITICAL: reloader spawns a second process
        ),
        daemon = True,
        name   = "FlaskServer",
    )
    flask_thread.start()
    return flask_thread


# ─────────────────────────────────────────────────────────────────────────────
# Main demo controller
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(
    scenarios : List[str],
    dwell     : float,
    interval  : float,
    no_browser: bool,
) -> None:
    """
    Main orchestration loop.

    WHY dwell time is configurable?
    --------------------------------
    --dwell 60  gives an examiner a slow, readable demo with many frames.
    --dwell 5   gives CI pipelines a fast smoke-test (< 30 s total).
    The default of 30 s balances readability vs demo length.

    WHY all 4 scenarios are cycled automatically?
    -----------------------------------------------
    An unattended run is suitable for a recorded video submission and
    proves the system responds correctly to every fault type without
    requiring manual intervention between scenarios.
    """

    # ── Step 1: Train ML models ──────────────────────────────────────────────
    print()
    print("═" * 60)
    print("  Smart Industrial Automation — Integration Demo")
    print("═" * 60)
    print()
    print("[DEMO] Step 1/3 — Training Isolation Forest models …")
    detector1.train(n_samples=200)
    detector2.train(n_samples=200)
    print("[DEMO] Both models trained.\n")

    # ── Step 2: Start Flask ──────────────────────────────────────────────────
    print("[DEMO] Step 2/3 — Starting Flask REST API …")
    _start_flask()
    time.sleep(1.0)   # grace period for Flask to bind to port 5000
    print("[DEMO] Flask live at http://localhost:5000")
    print("[DEMO] Endpoints: /api/status  /api/motor/1  /api/motor/2  "
          "/api/history  /api/alerts")

    # ── Step 2b: Open dashboard ──────────────────────────────────────────────
    dashboard_path = os.path.abspath("dashboard.html")
    if not no_browser:
        if os.path.exists(dashboard_path):
            webbrowser.open("file://" + dashboard_path)
            print(f"[DEMO] Dashboard opened: file://{dashboard_path}")
        else:
            print("[DEMO] dashboard.html not found — skipping auto-open.")
    print(f"[DEMO] API status: http://localhost:5000/api/status\n")

    # ── Step 3: Scenario sequencer ───────────────────────────────────────────
    print("[DEMO] Step 3/3 — Running fault scenario sequence …")

    summary_stats: List[Dict] = []
    total = len(scenarios)

    for sc_idx, scenario in enumerate(scenarios, start=1):
        fault_id   = FAULT_ID_MAP.get(scenario, 0)
        fault_name = FAULT_NAMES.get(fault_id, "UNKNOWN")

        _print_banner(sc_idx, total, scenario, dwell, interval)

        # Generate full sensor DataFrame for this scenario
        # duration_s=10, fs=1000 → 10,000 rows; we cycle through them
        df = generate_scenario(scenario, duration_s=10, fs=1000, seed=42)
        n_rows      = len(df)
        frames_sent = 0
        scores_m1   : List[float] = []
        scores_m2   : List[float] = []
        peak_status = "NORMAL"
        STATUS_RANK = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2, "INITIALISING": -1}

        deadline = time.monotonic() + dwell

        row_idx = 0
        while time.monotonic() < deadline:
            row = df.iloc[row_idx % n_rows]

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # ── Motor 1 readings ──────────────────────────────────────────
            temp_m1  = float(row["temperature_C"])
            vib_m1   = float(row["vibration_x_g"])
            speed_m1 = float(row["speed_rpm"])

            # Secondary (ambient) — simulated as temp−40 + humidity 55%
            amb_m1 = temp_m1 - 40.0
            hum_m1 = 55.0 + (fault_id * 2.0)   # slight humidity rise with faults

            _handle_primary(
                "motor1", "history1", detector1,
                temp_m1, vib_m1, speed_m1, fault_id, ts, is_motor2=False
            )
            _handle_secondary_state("motor1", amb_m1, hum_m1)

            # Capture score for summary
            with _lock:
                sc_m1  = state["motor1"].get("anomaly_score", 0.0) or 0.0
                st_m1  = state["motor1"].get("anomaly_status", "NORMAL")
            scores_m1.append(sc_m1)
            if STATUS_RANK.get(st_m1, 0) > STATUS_RANK.get(peak_status, 0):
                peak_status = st_m1

            # ── Motor 2 readings (pump — can_node.py scaling applied) ─────
            # Motor 2 runs cooler (×0.97), slightly rougher vib (×1.05),
            # speed → flow rate (÷14.8 L/min per RPM)
            temp_m2  = temp_m1  * 0.97
            vib_m2   = abs(vib_m1) * 1.05
            flow_m2  = speed_m1 / 14.8

            amb_m2 = temp_m2 - 40.0
            hum_m2 = 57.0 + (fault_id * 1.5)

            _handle_primary(
                "motor2", "history2", detector2,
                temp_m2, vib_m2, flow_m2, fault_id, ts, is_motor2=True
            )
            _handle_secondary_state("motor2", amb_m2, hum_m2)

            with _lock:
                sc_m2 = state["motor2"].get("anomaly_score", 0.0) or 0.0

            scores_m2.append(sc_m2)

            # ── Console output ────────────────────────────────────────────
            _print_frame_line(ts, "Motor1", temp_m1, vib_m1, speed_m1,
                              sc_m1, st_m1, fault_name, is_motor2=False)

            frames_sent += 1
            row_idx     += 1
            time.sleep(interval)

        # Scenario complete — record stats
        avg_score = (sum(scores_m1) / len(scores_m1)) if scores_m1 else 0.0
        summary_stats.append({
            "scenario"   : scenario[:12],
            "frames"     : frames_sent,
            "avg_score"  : avg_score,
            "peak_status": peak_status,
        })
        print(f"\n  [DEMO] Scenario '{scenario}' complete — "
              f"{frames_sent} frames, avg score {avg_score:+.4f}, "
              f"peak status: {peak_status}")

    # ── Final summary table ───────────────────────────────────────────────────
    _print_summary(summary_stats)

    # ── Keep Flask alive for examiner inspection ──────────────────────────────
    # WHY keep Flask running after the demo?
    # The examiner can query /api/status, /api/alerts etc. in a browser or
    # with curl to verify the API contract without needing to re-run the demo.
    # Only Ctrl-C terminates the process.
    print("[DEMO] Keeping API live.  Open http://localhost:5000/api/status "
          "in a browser.")
    print("[DEMO] Or run:  python demo_report.py   to generate an evidence report.")
    print("[DEMO] Press Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart Industrial Automation — Integration & Demo Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        metavar="SEC",
        help="Seconds between frame transmissions",
    )
    parser.add_argument(
        "--dwell", type=float, default=30.0,
        metavar="SEC",
        help="Seconds to spend on each scenario  "
             "(use --dwell 5 for CI, --dwell 60 for slow examiner demo)",
    )
    parser.add_argument(
        "--scenarios", type=str,
        default=",".join(SCENARIOS),
        metavar="LIST",
        help='Comma-separated subset, e.g. "normal,bearing_fault"',
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Skip auto-opening the dashboard in a browser tab",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Validate scenario list
    requested = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    invalid   = [s for s in requested if s not in SCENARIOS]
    if invalid:
        print(f"[ERROR] Unknown scenarios: {invalid}")
        print(f"        Valid choices: {list(SCENARIOS)}")
        sys.exit(1)
    if not requested:
        print("[ERROR] --scenarios list is empty.")
        sys.exit(1)

    try:
        run_demo(
            scenarios  = requested,
            dwell      = args.dwell,
            interval   = args.interval,
            no_browser = args.no_browser,
        )
    except KeyboardInterrupt:
        print("\n[DEMO] Shutting down cleanly.")
        sys.exit(0)

# python run_demo.py --dwell 30 --interval 0.5
if __name__ == "__main__":
    main()