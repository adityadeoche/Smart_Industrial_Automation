"""
can_node.py — CAN FD Transmitter Node
======================================
Simulates an ESP32 WROOM-32 CAN FD node that reads sensor data from
sensor_simulator.py and transmits it over a python-can virtual CAN FD bus.

CAN FD vs Classic CAN — KEY CONCEPT:
  Classic CAN : max 8 bytes payload per frame
  CAN FD      : max 64 bytes payload per frame (Flexible Data-rate)
  CAN FD also supports a higher bit-rate during the data phase (data_bitrate),
  while keeping a lower bit-rate during the arbitration phase (bitrate).
  This lets us pack all four float sensor values (16 bytes) into ONE frame
  instead of splitting across two classic CAN frames.

Arbitration ID — KEY CONCEPT:
  Every CAN frame carries an 11-bit (standard) or 29-bit (extended) ID.
  The bus arbitrates by ID: lower ID wins if two nodes transmit simultaneously.
  We use:
    0x100 / 0x101  →  Motor 1 primary / secondary data
    0x200 / 0x201  →  Motor 2 (pump) primary / secondary data
  The MSB difference (1xx vs 2xx) cleanly separates the two physical nodes,
  and a receiver can filter by ID range to listen to only one motor.

struct little-endian — KEY CONCEPT:
  ESP32 is a little-endian CPU (Xtensa LX6).
  struct.pack('<ffff', ...) packs four IEEE-754 floats in little-endian byte
  order so the ESP32 can memcpy() the raw bytes directly into a C float[4]
  without any byte-swapping. Using big-endian '>' would silently corrupt values
  on the ESP32 side.

is_fd=True — KEY CONCEPT:
  Tells python-can to build a CAN FD frame with the FDF (FD Format) bit set in
  the control field. A classic CAN transceiver will refuse/ignore this frame.
  Combined with data_bitrate=2_000_000, the data phase runs at 2 Mbit/s while
  the arbitration phase runs at 500 kbit/s — giving backward-compatible
  arbitration but 4× faster payload transfer.
"""

import argparse
import struct
import time
import sys
import itertools
from datetime import datetime
 
# ── python-can ────────────────────────────────────────────────────────────────
try:
    import can
except ImportError:
    print("[ERROR] python-can not installed.  Run:  pip install python-can")
    sys.exit(1)
 
# ── sensor_simulator — use its exact public API ───────────────────────────────
try:
    from sensor_simulator import generate_scenario, SCENARIOS
except ImportError as e:
    print(f"[ERROR] Cannot import sensor_simulator: {e}")
    print("        Ensure sensor_simulator.py is in the same folder as can_node.py")
    sys.exit(1)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CAN FD Arbitration IDs
# ─────────────────────────────────────────────────────────────────────────────
FRAME_MOTOR1_PRIMARY   = 0x100   # temp, vibration_x, speed_rpm,  fault_id
FRAME_MOTOR1_SECONDARY = 0x101   # ambient_temp, humidity
FRAME_MOTOR2_PRIMARY   = 0x200   # temp, vibration_x, flow_rate,  fault_id
FRAME_MOTOR2_SECONDARY = 0x201   # ambient_temp, humidity
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Struct format strings — little-endian ('<') for ESP32 compatibility
# ─────────────────────────────────────────────────────────────────────────────
#
# PRIMARY frame  (16 bytes total — next valid CAN FD DLC above 13):
#   '<fffBxxx'
#   Bytes  0– 3 : temperature_C   (float32, °C)
#   Bytes  4– 7 : vibration_x_g   (float32, g)
#   Bytes  8–11 : speed_rpm / flow_rate  (float32)
#   Byte  12    : fault_id         (uint8,  0–3)
#   Bytes 13–15 : 0x00 padding     (CAN FD DLC: 13 is invalid → pad to 16)
#
# SECONDARY frame (8 bytes — valid in both CAN FD and classic CAN):
#   '<ff'
#   Bytes 0–3 : ambient_temp  (float32, °C)
#   Bytes 4–7 : humidity      (float32, %)
#
PRIMARY_FMT   = '<fffBxxx'   # 16 bytes
SECONDARY_FMT = '<ff'        # 8 bytes
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Fault ID mapping — scenario string → uint8 sent on CAN bus
# ─────────────────────────────────────────────────────────────────────────────
FAULT_ID_MAP = {
    "normal"         : 0,
    "bearing_fault"  : 1,
    "stator_fault"   : 2,
    "rotor_bar_fault": 3,
}
 
# CLI alias → simulator scenario string
CLI_ALIAS = {
    "none"          : "normal",
    "normal"        : "normal",
    "bearing_wear"  : "bearing_fault",
    "bearing_fault" : "bearing_fault",
    "overheating"   : "stator_fault",
    "stator_fault"  : "stator_fault",
    "motor_overload": "rotor_bar_fault",
    "rotor_bar_fault": "rotor_bar_fault",
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Frame builders
# ─────────────────────────────────────────────────────────────────────────────
 
def build_primary_frame(arb_id: int, temp: float, vibration: float,
                        variable: float, fault_id: int) -> can.Message:
    """
    Pack 3 float sensor values + 1 fault code into a 16-byte CAN FD frame.
 
    struct format '<fffBxxx':
        '<'  = little-endian (ESP32 native byte order)
        'f'  = IEEE-754 float32, 4 bytes  (temp)
        'f'  = IEEE-754 float32, 4 bytes  (vibration)
        'f'  = IEEE-754 float32, 4 bytes  (RPM or flow_rate)
        'B'  = unsigned uint8,   1 byte   (fault_id)
        'xxx'= 3 padding bytes            (reach 16-byte DLC boundary)
    Total: 3×4 + 1 + 3 = 16 bytes
    """
    payload = struct.pack(
        PRIMARY_FMT,
        float(temp),
        float(vibration),
        float(variable),
        int(fault_id) & 0xFF,
    )
    return can.Message(
        arbitration_id=arb_id,
        data=payload,
        is_fd=True,          # FDF bit set → CAN FD frame, DLC > 8 allowed
        is_extended_id=False, # Standard 11-bit arbitration ID
    )
 
 
def build_secondary_frame(arb_id: int, ambient_temp: float,
                          humidity: float) -> can.Message:
    """
    Pack ambient_temp + humidity into an 8-byte CAN FD frame.
 
    struct format '<ff':  2 × float32 = 8 bytes  (also fits classic CAN)
    """
    payload = struct.pack(SECONDARY_FMT, float(ambient_temp), float(humidity))
    return can.Message(
        arbitration_id=arb_id,
        data=payload,
        is_fd=True,
        is_extended_id=False,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Console logger
# ─────────────────────────────────────────────────────────────────────────────
 
def log_frame(msg: can.Message, label: str, fields: dict):
    """Pretty-print a transmitted CAN FD frame with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    field_str = "  ".join(f"{k}={v}" for k, v in fields.items())
    print(f"  [{ts}] TX  ID=0x{msg.arbitration_id:03X}  {label:<28}  {field_str}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Derived-value helpers (fill columns the simulator doesn't produce)
# ─────────────────────────────────────────────────────────────────────────────
 
def derive_ambient_temp(temperature_C: float) -> float:
    """
    Ambient ≈ motor temp minus typical 38 °C winding/bearing rise.
    Clamped to realistic range [10, 45] °C.
    """
    return float(max(10.0, min(45.0, temperature_C - 38.0)))
 
 
def derive_humidity(time_s: float) -> float:
    """Simulate realistic indoor humidity with slow sinusoidal drift."""
    import math
    return 50.0 + 5.0 * math.sin(2 * math.pi * 0.02 * time_s)
 
 
def derive_flow_rate(speed_rpm: float) -> float:
    """
    Pump affinity law: flow ∝ speed.
    At rated 1480 RPM → 100 L/min nominal.
    """
    return float(max(0.0, speed_rpm / 14.8))
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main transmit loop
# ─────────────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="CAN FD transmitter node — simulates ESP32 WROOM-32"
    )
    parser.add_argument(
        "--fault",
        default="none",
        metavar="SCENARIO",
        help=(
            "Fault scenario to inject. "
            "Options: none, bearing_wear, overheating, motor_overload  "
            "(aliases for: normal, bearing_fault, stator_fault, rotor_bar_fault)"
        ),
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Transmit interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Sensor data generation duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--fs", type=float, default=1000.0,
        help="Sensor sampling frequency in Hz (default: 1000)",
    )
    args = parser.parse_args()
 
    # ── Resolve fault alias → simulator scenario string ───────────────────────
    fault_key = args.fault.lower().replace('-', '_')
    if fault_key not in CLI_ALIAS:
        print(f"[ERROR] Unknown fault '{args.fault}'.")
        print(f"        Valid options: {', '.join(CLI_ALIAS.keys())}")
        sys.exit(1)
 
    scenario = CLI_ALIAS[fault_key]   # e.g. "bearing_fault"
    fault_id = FAULT_ID_MAP[scenario] # e.g. 1
 
    print(f"\n{'='*68}")
    print(f"  CAN FD Node — python-can virtual bus")
    print(f"  Scenario    : {scenario}  (fault_id={fault_id})")
    print(f"  Interval    : {args.interval}s")
    print(f"  Bitrate     : 500 kbit/s  (arbitration phase)")
    print(f"  Data bitrate: 2 Mbit/s   (data phase, CAN FD BRS)")
    print(f"  Frames/cycle: 4  (0x100, 0x101, 0x200, 0x201)")
    print(f"{'='*68}\n")
 
    # ── Generate sensor data ──────────────────────────────────────────────────
    print(f"[INFO] Calling generate_scenario('{scenario}', "
          f"duration_s={args.duration}, fs={args.fs}) …")
    df = generate_scenario(scenario, duration_s=args.duration, fs=args.fs)
    print(f"[INFO] DataFrame shape : {df.shape}")
    print(f"[INFO] Columns         : {list(df.columns)}")
    print(f"[INFO] Rows will be streamed one-per-cycle (wraps at end)\n")
 
    # Cycle through DataFrame rows endlessly
    row_iter = itertools.cycle(df.itertuples(index=False))
 
    # ── Open virtual CAN FD bus ───────────────────────────────────────────────
    # 'virtual' interface works on Windows / macOS / Linux — no hardware needed.
    # For SocketCAN on Linux use: interface='socketcan', channel='vcan0'
    bus = can.Bus(
        interface = 'udp_multicast',
        channel   = '239.0.0.1',
        fd        = True,
    )
 
    print("Bus open. Transmitting… (Ctrl-C to stop)\n")
    print(f"  {'Timestamp':<15} {'Dir':<4} {'ID':<8} {'Frame':<28} Sensor values")
    print(f"  {'-'*100}")
 
    cycle = 0
    try:
        while True:
            cycle += 1
            row = next(row_iter)
 
            # ── Extract real columns from simulator DataFrame ──────────────────
            temp       = float(row.temperature_C)   # °C — bearing/winding temp
            vibration  = float(row.vibration_x_g)   # g  — radial X vibration
            speed      = float(row.speed_rpm)        # RPM
            time_s     = float(row.time_s)
 
            # ── Derive values not in the simulator ────────────────────────────
            amb_temp   = derive_ambient_temp(temp)
            humidity   = derive_humidity(time_s)
            flow_rate  = derive_flow_rate(speed)     # for pump (Motor 2)
 
            # Motor 2 (pump): slight variance from Motor 1 to simulate
            # two independent physical machines on the same CAN bus
            temp_m2      = temp      * 0.97          # pump runs ~3% cooler
            vibration_m2 = vibration * 1.05          # pump slightly rougher
            amb_temp_m2  = amb_temp  + 0.5
            humidity_m2  = humidity  - 1.0
 
            # ── Build 4 CAN FD frames ─────────────────────────────────────────
            frame_0x100 = build_primary_frame(
                FRAME_MOTOR1_PRIMARY, temp, vibration, speed, fault_id)
            frame_0x101 = build_secondary_frame(
                FRAME_MOTOR1_SECONDARY, amb_temp, humidity)
            frame_0x200 = build_primary_frame(
                FRAME_MOTOR2_PRIMARY, temp_m2, vibration_m2, flow_rate, fault_id)
            frame_0x201 = build_secondary_frame(
                FRAME_MOTOR2_SECONDARY, amb_temp_m2, humidity_m2)
 
            # ── Transmit all 4 frames and log to console ──────────────────────
            for frame, label, fields in [
                (frame_0x100, "Motor1 Primary", {
                    "temp": f"{temp:.1f}C",
                    "vib":  f"{vibration:.4f}g",
                    "rpm":  f"{speed:.1f}",
                    "fault": fault_id,
                }),
                (frame_0x101, "Motor1 Secondary", {
                    "amb": f"{amb_temp:.1f}C",
                    "hum": f"{humidity:.1f}%",
                }),
                (frame_0x200, "Motor2(Pump) Primary", {
                    "temp": f"{temp_m2:.1f}C",
                    "vib":  f"{vibration_m2:.4f}g",
                    "flow": f"{flow_rate:.1f}L/m",
                    "fault": fault_id,
                }),
                (frame_0x201, "Motor2(Pump) Secondary", {
                    "amb": f"{amb_temp_m2:.1f}C",
                    "hum": f"{humidity_m2:.1f}%",
                }),
            ]:
                bus.send(frame)
                log_frame(frame, label, fields)
 
            print(f"  {'─'*100}")
            time.sleep(args.interval)
 
    except KeyboardInterrupt:
        print("\n\n[INFO] Transmit stopped by user.")
    finally:
        bus.shutdown()
        print("[INFO] CAN bus shut down cleanly.")
 
 
if __name__ == "__main__":
    main()