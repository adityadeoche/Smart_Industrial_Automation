"""
test_canfd.py — CAN FD Frame Packing / Unpacking Integrity
==========================================================
Validates that sensor data is correctly serialised into CAN FD frames
by can_node.py and correctly deserialised by ml_gateway.py.

WHY float round-trip tolerance of 0.01?
  Struct format '<f' packs a Python float64 into IEEE-754 float32 (4 bytes).
  Float32 has ~7 significant decimal digits of precision.
  For sensor values in typical ranges:
    • temperature 65.0°C → quantisation error ≈ 65 × 2^-23 ≈ 7.7×10^-6 (< 0.01)
    • vibration 0.05g    → quantisation error ≈ 0.05 × 2^-23 ≈ 6×10^-9 (< 0.01)
    • speed 1480 RPM     → quantisation error ≈ 1480 × 2^-23 ≈ 1.76×10^-4 (< 0.01)
  The 0.01 tolerance safely covers worst-case float32 quantisation for all
  values in our operating range. Tighter (e.g. 1e-6) would fail for large
  values; looser (e.g. 1.0) would mask actual encoding bugs.

WHY NOT use the 'DLC' attribute for length check?
  python-can's can.Message.dlc is the CAN DLC code (0-15 for CAN FD) which
  maps non-linearly to byte count for DLC > 8. len(msg.data) directly returns
  the actual byte length of the data buffer — unambiguous and portable across
  python-can versions.

Industrial relevance:
  This test suite corresponds to FAT Level 2 — "Communication Interface Test".
  In a real deployment, an integrator would verify frame byte-order and field
  mapping on the oscilloscope before connecting the ESP32 to the Raspberry Pi.
  Here, the struct round-trip test replaces the oscilloscope with a software
  assertion, providing the same quantitative guarantee in automated form.
"""

import struct
import pytest

from can_node import (
    build_primary_frame,
    build_secondary_frame,
    PRIMARY_FMT,
    SECONDARY_FMT,
    FAULT_ID_MAP,
    FRAME_MOTOR1_PRIMARY,
    FRAME_MOTOR1_SECONDARY,
    FRAME_MOTOR2_PRIMARY,
    FRAME_MOTOR2_SECONDARY,
)
from ml_gateway import _unpack_primary, _unpack_secondary


# ─────────────────────────────────────────────────────────────────────────────
# 1. Struct format sizes
# ─────────────────────────────────────────────────────────────────────────────

class TestStructFormats:
    """
    Validates the byte-level frame layout against the hardware specification.

    PRIMARY_FMT = '<fffBxxx'  →  3×4 + 1 + 3 = 16 bytes
    SECONDARY_FMT = '<ff'     →  2×4 = 8 bytes

    CAN FD DLC validity:
      Valid DLC byte counts: 0,1,2,3,4,5,6,7,8,12,16,20,24,32,48,64.
      13 is NOT a valid DLC — the firmware must pad to 16 bytes.
      8 is valid for both classic CAN and CAN FD.
    """

    def test_primary_fmt_size_is_16(self):
        """PRIMARY_FMT must produce exactly 16 bytes — a valid CAN FD DLC."""
        assert struct.calcsize(PRIMARY_FMT) == 16, (
            f"PRIMARY_FMT '{PRIMARY_FMT}' calcsize = {struct.calcsize(PRIMARY_FMT)}, "
            f"expected 16 bytes"
        )

    def test_secondary_fmt_size_is_8(self):
        """SECONDARY_FMT must produce exactly 8 bytes — valid for both CAN FD and classic CAN."""
        assert struct.calcsize(SECONDARY_FMT) == 8, (
            f"SECONDARY_FMT '{SECONDARY_FMT}' calcsize = {struct.calcsize(SECONDARY_FMT)}, "
            f"expected 8 bytes"
        )

    def test_primary_fmt_is_little_endian(self):
        """Format string must start with '<' (little-endian for ESP32 Xtensa LX6)."""
        assert PRIMARY_FMT.startswith("<"), (
            f"PRIMARY_FMT '{PRIMARY_FMT}' not little-endian ('<')"
        )

    def test_secondary_fmt_is_little_endian(self):
        assert SECONDARY_FMT.startswith("<"), (
            f"SECONDARY_FMT '{SECONDARY_FMT}' not little-endian ('<')"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frame flags — is_fd and is_extended_id
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameFlags:
    """
    Every frame — primary and secondary, Motor 1 and Motor 2 — must:
      • Set is_fd=True   (FDF bit in CAN control field → CAN FD frame)
      • Set is_extended_id=False  (11-bit standard arbitration ID, not 29-bit)

    WHY is_fd must be True?
      Classic CAN limits payload to 8 bytes. Our PRIMARY frame is 16 bytes.
      Without is_fd=True, the bus controller raises a DLC error on any
      payload > 8 bytes. The FDF bit tells transceivers this is a CAN FD frame
      and to accept up to 64-byte payloads.

    WHY is_extended_id must be False?
      Extended IDs (29-bit) add 18 extra bits to the arbitration field,
      increasing minimum frame overhead. Standard 11-bit IDs (0x000–0x7FF)
      suffice for our 4 frame types and give higher priority resolution.
    """

    ARBS = [
        (FRAME_MOTOR1_PRIMARY,   0x100, "primary",   "motor1"),
        (FRAME_MOTOR1_SECONDARY, 0x101, "secondary", "motor1"),
        (FRAME_MOTOR2_PRIMARY,   0x200, "primary",   "motor2"),
        (FRAME_MOTOR2_SECONDARY, 0x201, "secondary", "motor2"),
    ]

    @pytest.mark.parametrize("arb_id,expected_id,ftype,motor", ARBS)
    def test_is_fd_flag(self, arb_id, expected_id, ftype, motor):
        if ftype == "primary":
            msg = build_primary_frame(arb_id, 65.0, 0.05, 1480.0, 0)
        else:
            msg = build_secondary_frame(arb_id, 27.0, 50.0)
        assert msg.is_fd is True, (
            f"Frame 0x{arb_id:03X} ({motor} {ftype}) is_fd=False — "
            "must be True for CAN FD payload > 8 bytes or to set FDF bit"
        )

    @pytest.mark.parametrize("arb_id,expected_id,ftype,motor", ARBS)
    def test_is_extended_id_false(self, arb_id, expected_id, ftype, motor):
        if ftype == "primary":
            msg = build_primary_frame(arb_id, 65.0, 0.05, 1480.0, 0)
        else:
            msg = build_secondary_frame(arb_id, 27.0, 50.0)
        assert msg.is_extended_id is False, (
            f"Frame 0x{arb_id:03X} is_extended_id=True — "
            "must use standard 11-bit arbitration ID"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Arbitration IDs
# ─────────────────────────────────────────────────────────────────────────────

class TestArbitrationIDs:
    """
    Verifies that each frame builder produces the correct arbitration ID.
    ID mismatch would cause the receiver to route the frame to the wrong
    motor or silently discard it.
    """

    def test_motor1_primary_id(self):
        msg = build_primary_frame(FRAME_MOTOR1_PRIMARY, 65.0, 0.05, 1480.0, 0)
        assert msg.arbitration_id == 0x100

    def test_motor1_secondary_id(self):
        msg = build_secondary_frame(FRAME_MOTOR1_SECONDARY, 27.0, 50.0)
        assert msg.arbitration_id == 0x101

    def test_motor2_primary_id(self):
        msg = build_primary_frame(FRAME_MOTOR2_PRIMARY, 63.0, 0.05, 100.0, 0)
        assert msg.arbitration_id == 0x200

    def test_motor2_secondary_id(self):
        msg = build_secondary_frame(FRAME_MOTOR2_SECONDARY, 27.5, 49.0)
        assert msg.arbitration_id == 0x201

    def test_custom_id_passthrough(self):
        """
        build_primary_frame must accept ANY valid 11-bit ID (0x000–0x7FF).
        """
        for arb_id in [0x000, 0x100, 0x200, 0x7FF]:
            msg = build_primary_frame(arb_id, 65.0, 0.05, 1480.0, 0)
            assert msg.arbitration_id == arb_id


# ─────────────────────────────────────────────────────────────────────────────
# 4. Data length (byte count, not DLC code)
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLength:
    """
    Use len(msg.data) — NOT msg.dlc — to check payload byte count.
    For CAN FD, dlc codes 9-15 map non-linearly: DLC=9→12 bytes, DLC=10→16 bytes.
    len(msg.data) is unambiguous.
    """

    def test_primary_frame_length_16(self):
        for arb_id in [FRAME_MOTOR1_PRIMARY, FRAME_MOTOR2_PRIMARY]:
            msg = build_primary_frame(arb_id, 65.0, 0.05, 1480.0, 0)
            assert len(msg.data) == 16, (
                f"Frame 0x{arb_id:03X} data length {len(msg.data)} != 16 bytes"
            )

    def test_secondary_frame_length_8(self):
        for arb_id in [FRAME_MOTOR1_SECONDARY, FRAME_MOTOR2_SECONDARY]:
            msg = build_secondary_frame(arb_id, 27.0, 50.0)
            assert len(msg.data) == 8, (
                f"Frame 0x{arb_id:03X} data length {len(msg.data)} != 8 bytes"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Round-trip tests — pack → unpack → compare
# ─────────────────────────────────────────────────────────────────────────────

FLOAT_TOLERANCE = 0.01   # IEEE-754 float32 quantisation bound


class TestPrimaryRoundTrip:
    """
    Verifies that build_primary_frame() → _unpack_primary() is lossless
    within float32 quantisation limits.

    Test values cover the full expected operating range:
      • Normal: temp=65, vib=0.05, rpm=1480
      • Extremes: temp=90 (stator fault), vib=0.5 (bearing fault), rpm=1465
    """

    NORMAL_CASES = [
        (0x100, 65.0,  0.05,   1480.0, 0),
        (0x100, 80.0,  0.35,   1470.0, 1),   # bearing fault values
        (0x100, 90.0,  0.12,   1460.0, 2),   # stator fault values
        (0x100, 73.0,  0.08,   1465.0, 3),   # rotor bar values
        (0x200, 63.0,  0.0525, 100.0,  0),   # Motor 2 (pump, flow in L/min)
        (0x200, 77.0,  0.368,  97.0,   1),   # Motor 2 bearing fault
    ]

    @pytest.mark.parametrize("arb_id,temp,vib,variable,fault_id", NORMAL_CASES)
    def test_round_trip_temperature(self, arb_id, temp, vib, variable, fault_id):
        msg = build_primary_frame(arb_id, temp, vib, variable, fault_id)
        t_out, _, _, _ = _unpack_primary(msg.data)
        assert abs(t_out - temp) < FLOAT_TOLERANCE, (
            f"Temperature round-trip error: packed {temp}, got {t_out}"
        )

    @pytest.mark.parametrize("arb_id,temp,vib,variable,fault_id", NORMAL_CASES)
    def test_round_trip_vibration(self, arb_id, temp, vib, variable, fault_id):
        msg = build_primary_frame(arb_id, temp, vib, variable, fault_id)
        _, v_out, _, _ = _unpack_primary(msg.data)
        assert abs(v_out - vib) < FLOAT_TOLERANCE, (
            f"Vibration round-trip error: packed {vib}, got {v_out}"
        )

    @pytest.mark.parametrize("arb_id,temp,vib,variable,fault_id", NORMAL_CASES)
    def test_round_trip_variable(self, arb_id, temp, vib, variable, fault_id):
        msg = build_primary_frame(arb_id, temp, vib, variable, fault_id)
        _, _, var_out, _ = _unpack_primary(msg.data)
        assert abs(var_out - variable) < FLOAT_TOLERANCE, (
            f"Variable round-trip error: packed {variable}, got {var_out}"
        )

    @pytest.mark.parametrize("arb_id,temp,vib,variable,fault_id", NORMAL_CASES)
    def test_round_trip_fault_id_exact(self, arb_id, temp, vib, variable, fault_id):
        """
        fault_id is a uint8 packed with 'B' — no floating-point quantisation.
        Round-trip must be EXACT (zero tolerance).
        """
        msg = build_primary_frame(arb_id, temp, vib, variable, fault_id)
        _, _, _, fid_out = _unpack_primary(msg.data)
        assert fid_out == fault_id, (
            f"fault_id round-trip: packed {fault_id}, got {fid_out}"
        )


class TestSecondaryRoundTrip:
    """
    Round-trip tests for the SECONDARY frame (ambient_temp + humidity).
    """

    SECONDARY_CASES = [
        (0x101, 27.0,  50.0),
        (0x101, 10.0,  30.0),   # cold-dry extreme
        (0x101, 45.0,  80.0),   # hot-humid extreme
        (0x201, 27.5,  49.0),   # Motor 2 secondary
        (0x201, 15.3,  62.5),
    ]

    @pytest.mark.parametrize("arb_id,amb_temp,humidity", SECONDARY_CASES)
    def test_round_trip_ambient_temp(self, arb_id, amb_temp, humidity):
        msg = build_secondary_frame(arb_id, amb_temp, humidity)
        amb_out, _ = _unpack_secondary(msg.data)
        assert abs(amb_out - amb_temp) < FLOAT_TOLERANCE, (
            f"Ambient temp round-trip: packed {amb_temp}, got {amb_out}"
        )

    @pytest.mark.parametrize("arb_id,amb_temp,humidity", SECONDARY_CASES)
    def test_round_trip_humidity(self, arb_id, amb_temp, humidity):
        msg = build_secondary_frame(arb_id, amb_temp, humidity)
        _, hum_out = _unpack_secondary(msg.data)
        assert abs(hum_out - humidity) < FLOAT_TOLERANCE, (
            f"Humidity round-trip: packed {humidity}, got {hum_out}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fault ID encoding — all four fault codes
# ─────────────────────────────────────────────────────────────────────────────

class TestFaultIDEncoding:
    """
    Validates the FAULT_ID_MAP values and their round-trip integrity.

    The fault_id byte is the ONLY non-float field in the PRIMARY frame.
    It acts as a ground-truth label that allows the Raspberry Pi gateway
    to cross-validate the ML model's anomaly classification against the
    ground-truth injected fault — a key data point for the research paper.
    """

    @pytest.mark.parametrize("scenario,expected_id", [
        ("normal",          0),
        ("bearing_fault",   1),
        ("stator_fault",    2),
        ("rotor_bar_fault", 3),
    ])
    def test_fault_id_map_values(self, scenario, expected_id):
        assert FAULT_ID_MAP[scenario] == expected_id, (
            f"FAULT_ID_MAP['{scenario}'] = {FAULT_ID_MAP[scenario]}, "
            f"expected {expected_id}"
        )

    @pytest.mark.parametrize("scenario,expected_id", [
        ("normal",          0),
        ("bearing_fault",   1),
        ("stator_fault",    2),
        ("rotor_bar_fault", 3),
    ])
    def test_fault_id_round_trip(self, scenario, expected_id):
        """Pack fault_id for each scenario and verify exact unpack."""
        fault_id = FAULT_ID_MAP[scenario]
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0, fault_id)
        _, _, _, fid_out = _unpack_primary(msg.data)
        assert fid_out == expected_id, (
            f"Fault ID round-trip failed for '{scenario}': "
            f"packed {fault_id}, got {fid_out}"
        )

    def test_fault_id_boundary_255(self):
        """
        uint8 boundary test — fault_id=255 must survive the AND mask (& 0xFF).
        Values above 3 are invalid in production but must not crash the packer.
        """
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0, 255)
        _, _, _, fid_out = _unpack_primary(msg.data)
        assert fid_out == 255

    def test_fault_id_boundary_0(self):
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0, 0)
        _, _, _, fid_out = _unpack_primary(msg.data)
        assert fid_out == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Error handling — unpack with truncated data
# ─────────────────────────────────────────────────────────────────────────────

class TestUnpackErrorHandling:
    """
    Validates that _unpack_primary and _unpack_secondary raise ValueError
    when given truncated data — not silent corruption.
    This mirrors the gateway's defensive programming against corrupted CAN frames.
    """

    def test_unpack_primary_too_short_raises(self):
        short_data = bytes(10)   # 10 < 16 (PRIMARY_SIZE)
        with pytest.raises(ValueError, match="too short"):
            _unpack_primary(short_data)

    def test_unpack_secondary_too_short_raises(self):
        short_data = bytes(4)   # 4 < 8 (SECONDARY_SIZE)
        with pytest.raises(ValueError, match="too short"):
            _unpack_secondary(short_data)

    def test_unpack_primary_accepts_exact_size(self):
        """16 bytes is exactly PRIMARY_SIZE — must not raise."""
        msg = build_primary_frame(0x100, 65.0, 0.05, 1480.0, 0)
        result = _unpack_primary(bytes(msg.data))
        assert len(result) == 4   # (temp, vib, variable, fault_id)

    def test_unpack_secondary_accepts_exact_size(self):
        """8 bytes is exactly SECONDARY_SIZE — must not raise."""
        msg = build_secondary_frame(0x101, 27.0, 50.0)
        result = _unpack_secondary(bytes(msg.data))
        assert len(result) == 2   # (ambient_temp, humidity)

    def test_unpack_primary_accepts_longer_data(self):
        """
        unpack_from is used (not unpack), so extra bytes at the end are
        silently ignored — this is intentional for future frame expansion.
        """
        extra_data = bytes(20)   # 20 > 16
        result = _unpack_primary(extra_data)
        assert len(result) == 4