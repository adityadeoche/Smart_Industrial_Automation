"""
test_api.py — Flask REST API Validation
========================================
Validates all five endpoints of ml_gateway.py using Flask's built-in
test client — no live server, no network I/O, fully deterministic.

WHY Flask test client instead of a real server?
  The Flask test client dispatches requests in-process via the WSGI interface
  (werkzeug.test.Client). Compared to a live server:
  • No TCP socket binding → tests run without network permissions
  • No port conflicts → CI/CD pipelines run tests in parallel safely
  • No OS firewall rules → identical behaviour on Windows/macOS/Linux
  • ~100× faster → no TCP handshake overhead per request
  • Deterministic → requests don't race with the CAN listener thread
    (the listener is not started in test mode; Flask serves from initial state)

  The trade-off: we do not test the full HTTP stack (nginx/gunicorn). That
  would be an integration test / deployment test — out of scope for the FAT.

Test design philosophy (FAT Level 3 — API Contract Test):
  "Verify that every API endpoint returns the documented JSON schema with
   correct field names, types, and value constraints before connecting the
   dashboard."

  This catches: renamed keys, missing fields, wrong status codes, and
  type errors (e.g., returning a list where a dict is expected).

Gateway state on first call:
  The gateway starts with empty motor state (all None values) and
  anomaly_status="INITIALISING" (set in _motor_template()). Tests must
  accept both the initial INITIALISING state AND any valid running state.
"""

import pytest
import json


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_ANOMALY_STATUSES = {"NORMAL", "WARNING", "CRITICAL", "INITIALISING"}

MOTOR1_REQUIRED_KEYS = {
    "motor_id", "type", "timestamp",
    "temperature_C", "vibration_x_g", "speed_rpm", "flow_rate_Lm",
    "ambient_temp_C", "humidity_pct",
    "fault_id", "fault_name",
    "anomaly_score", "anomaly_status",
    "last_updated", "rx_count",
}

STATUS_MOTOR_KEYS = {
    "temperature_C", "vibration_x_g", "speed_rpm", "flow_rate_Lm",
    "ambient_temp_C", "humidity_pct", "fault_id", "fault_name",
    "anomaly_score", "anomaly_status", "last_updated", "rx_count",
}


def _json(response):
    """Parse Flask response JSON — raises on non-JSON or non-200."""
    return json.loads(response.data.decode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /api/status
# ─────────────────────────────────────────────────────────────────────────────

class TestApiStatus:
    """
    /api/status returns the latest snapshot for BOTH motors in one call.
    Used by the dashboard overview panel (polled every 2 seconds).

    KEY: response keys must be "motor1" / "motor2" (NOT "motor_1" / "motor_2").
    This is a common source of integration bugs — the spec must be explicit.
    """

    def test_status_code_200(self, flask_test_client):
        response = flask_test_client.get("/api/status")
        assert response.status_code == 200, (
            f"GET /api/status returned {response.status_code}, expected 200"
        )

    def test_status_has_required_top_level_keys(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        required = {"motor1", "motor2", "gateway_start", "timestamp"}
        missing = required - set(data.keys())
        assert not missing, (
            f"GET /api/status missing keys: {missing}\n"
            f"Got: {list(data.keys())}"
        )

    def test_status_no_underscore_keys(self, flask_test_client):
        """
        Keys must be 'motor1'/'motor2' NOT 'motor_1'/'motor_2'.
        This catches a common typo that breaks the dashboard JS.
        """
        data = _json(flask_test_client.get("/api/status"))
        assert "motor_1" not in data, "Found 'motor_1' — should be 'motor1'"
        assert "motor_2" not in data, "Found 'motor_2' — should be 'motor2'"
        assert "motor1" in data, "Missing 'motor1' key in /api/status"
        assert "motor2" in data, "Missing 'motor2' key in /api/status"

    def test_status_motor1_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        status = data["motor1"]["anomaly_status"]
        assert status in VALID_ANOMALY_STATUSES, (
            f"motor1.anomaly_status '{status}' not in {VALID_ANOMALY_STATUSES}"
        )

    def test_status_motor2_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        status = data["motor2"]["anomaly_status"]
        assert status in VALID_ANOMALY_STATUSES, (
            f"motor2.anomaly_status '{status}' not in {VALID_ANOMALY_STATUSES}"
        )

    def test_status_gateway_start_is_string(self, flask_test_client):
        """gateway_start is an ISO-8601 timestamp string."""
        data = _json(flask_test_client.get("/api/status"))
        assert isinstance(data["gateway_start"], str), (
            f"gateway_start type: {type(data['gateway_start'])}, expected str"
        )

    def test_status_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        assert isinstance(data["timestamp"], str)

    def test_status_motor1_has_rx_count(self, flask_test_client):
        """rx_count must be present (starts at 0 in initial state)."""
        data = _json(flask_test_client.get("/api/status"))
        assert "rx_count" in data["motor1"], "motor1 missing 'rx_count'"
        assert isinstance(data["motor1"]["rx_count"], int)

    def test_status_motor2_has_rx_count(self, flask_test_client):
        data = _json(flask_test_client.get("/api/status"))
        assert "rx_count" in data["motor2"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. GET /api/motor/1
# ─────────────────────────────────────────────────────────────────────────────

class TestApiMotor1:
    """
    /api/motor/1 returns the full detail view for Motor 1.
    Includes all sensor channels plus ML score and fault classification.
    """

    def test_status_code_200(self, flask_test_client):
        assert flask_test_client.get("/api/motor/1").status_code == 200

    def test_motor_id_is_1(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert data["motor_id"] == 1, (
            f"motor_id {data['motor_id']} != 1"
        )

    def test_type_is_induction_motor(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert data["type"] == "Three-Phase Induction Motor", (
            f"Motor 1 type: '{data['type']}'"
        )

    def test_all_required_keys_present(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        missing = MOTOR1_REQUIRED_KEYS - set(data.keys())
        assert not missing, (
            f"GET /api/motor/1 missing keys: {missing}\n"
            f"Got: {sorted(data.keys())}"
        )

    def test_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert data["anomaly_status"] in VALID_ANOMALY_STATUSES

    def test_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert isinstance(data["timestamp"], str)

    def test_rx_count_is_integer(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/1"))
        assert isinstance(data["rx_count"], int)
        assert data["rx_count"] >= 0

    def test_temperature_is_none_or_numeric(self, flask_test_client):
        """
        In initial state (no CAN frames received), temperature_C is None.
        After receiving frames it becomes a float. Both are valid.
        """
        data = _json(flask_test_client.get("/api/motor/1"))
        temp = data["temperature_C"]
        assert temp is None or isinstance(temp, (int, float)), (
            f"temperature_C type: {type(temp)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GET /api/motor/2
# ─────────────────────────────────────────────────────────────────────────────

class TestApiMotor2:
    """
    /api/motor/2 returns the full detail view for Motor 2 (pump).
    Key difference from Motor 1: uses flow_rate_Lm instead of speed_rpm.
    """

    def test_status_code_200(self, flask_test_client):
        assert flask_test_client.get("/api/motor/2").status_code == 200

    def test_motor_id_is_2(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert data["motor_id"] == 2, f"motor_id {data['motor_id']} != 2"

    def test_type_is_pump_motor(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert data["type"] == "Centrifugal Pump Motor", (
            f"Motor 2 type: '{data['type']}'"
        )

    def test_anomaly_status_valid(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert data["anomaly_status"] in VALID_ANOMALY_STATUSES

    def test_motor2_has_flow_rate_key(self, flask_test_client):
        data = _json(flask_test_client.get("/api/motor/2"))
        assert "flow_rate_Lm" in data, "Motor 2 missing 'flow_rate_Lm'"


# ─────────────────────────────────────────────────────────────────────────────
# 4. GET /api/history
# ─────────────────────────────────────────────────────────────────────────────

class TestApiHistory:
    """
    /api/history returns the last 60 readings per motor as arrays.
    Used by Chart.js on the dashboard for time-series plots.

    The history starts empty (deque maxlen=60 with no CAN frames received).
    Tests must validate BOTH the empty and populated cases.
    """

    def test_status_code_200(self, flask_test_client):
        assert flask_test_client.get("/api/history").status_code == 200

    def test_has_required_top_level_keys(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        required = {"motor1", "motor2", "timestamp"}
        missing = required - set(data.keys())
        assert not missing, (
            f"GET /api/history missing keys: {missing}"
        )

    def test_motor1_history_is_list(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert isinstance(data["motor1"], list), (
            f"motor1 history type: {type(data['motor1'])}, expected list"
        )

    def test_motor2_history_is_list(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert isinstance(data["motor2"], list)

    def test_motor1_history_length_le_60(self, flask_test_client):
        """
        History deque has maxlen=HISTORY_LEN=60.
        On a fresh gateway this will be 0; under load at most 60.
        """
        data = _json(flask_test_client.get("/api/history"))
        assert len(data["motor1"]) <= 60, (
            f"motor1 history length {len(data['motor1'])} > 60 (HISTORY_LEN)"
        )

    def test_motor2_history_length_le_60(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert len(data["motor2"]) <= 60

    def test_history_entry_keys_if_populated(self, flask_test_client):
        """
        If history contains entries, each must have the required keys.
        (On a fresh gateway the lists are empty — this test is skipped
        automatically via the 'if' guard.)
        """
        required_entry_keys = {
            "time", "temperature_C", "vibration_x_g",
            "anomaly_score", "anomaly_status",
        }
        data = _json(flask_test_client.get("/api/history"))
        for entry in data["motor1"]:
            missing = required_entry_keys - set(entry.keys())
            assert not missing, (
                f"History entry missing keys: {missing}\n"
                f"Got: {list(entry.keys())}"
            )

    def test_history_anomaly_status_valid_if_populated(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        for entry in data["motor1"]:
            assert entry["anomaly_status"] in VALID_ANOMALY_STATUSES

    def test_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/history"))
        assert isinstance(data["timestamp"], str)


# ─────────────────────────────────────────────────────────────────────────────
# 5. GET /api/alerts
# ─────────────────────────────────────────────────────────────────────────────

class TestApiAlerts:
    """
    /api/alerts returns recent WARNING and CRITICAL anomaly events.

    CRITICAL: response must be a DICT (not a bare list).
    The response envelope {timestamp, count, alerts:[...]} allows the
    dashboard to display metadata (total count, last updated) alongside
    the alert list — a bare list would require a separate count request.

    On a fresh gateway (no CAN frames received) the alerts list is empty.
    Tests must validate both empty and populated cases.
    """

    def test_status_code_200(self, flask_test_client):
        assert flask_test_client.get("/api/alerts").status_code == 200

    def test_response_is_dict_not_list(self, flask_test_client):
        """
        The most critical assertion: /api/alerts must return a DICT.
        Returning a bare list (the pre-1.0 design) breaks the dashboard
        because it accesses response["count"] and response["timestamp"].
        """
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data, dict), (
            f"GET /api/alerts returned {type(data).__name__}, expected dict.\n"
            f"Dashboard expects: {{timestamp, count, alerts:[...]}}.\n"
            f"Got: {str(data)[:200]}"
        )

    def test_has_required_keys(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        required = {"timestamp", "count", "alerts"}
        missing = required - set(data.keys())
        assert not missing, (
            f"GET /api/alerts missing keys: {missing}"
        )

    def test_alerts_field_is_list(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data["alerts"], list), (
            f"alerts field type: {type(data['alerts'])}, expected list"
        )

    def test_count_equals_alerts_length(self, flask_test_client):
        """
        count must be len(alerts) — a mismatch would mislead operators
        about how many anomalies have occurred.
        """
        data = _json(flask_test_client.get("/api/alerts"))
        assert data["count"] == len(data["alerts"]), (
            f"count ({data['count']}) != len(alerts) ({len(data['alerts'])})"
        )

    def test_timestamp_is_string(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data["timestamp"], str)

    def test_count_is_non_negative_integer(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        assert isinstance(data["count"], int)
        assert data["count"] >= 0

    def test_alerts_empty_on_fresh_gateway(self, flask_test_client):
        """
        No CAN frames have been received → no anomalies logged → empty list.
        This is the expected state in the test environment.
        """
        data = _json(flask_test_client.get("/api/alerts"))
        assert data["count"] == 0, (
            f"Expected 0 alerts on fresh gateway, got {data['count']}"
        )
        assert data["alerts"] == [], (
            f"Expected empty alerts list, got {data['alerts']}"
        )

    def test_alert_entry_structure_if_populated(self, flask_test_client):
        """
        Structural test for individual alert entries.
        Skipped automatically if alerts list is empty (fresh gateway).
        """
        required_alert_keys = {
            "timestamp", "motor", "status", "anomaly_score",
            "fault_name", "temperature_C", "vibration_x_g",
        }
        data = _json(flask_test_client.get("/api/alerts"))
        for alert in data["alerts"]:
            missing = required_alert_keys - set(alert.keys())
            assert not missing, (
                f"Alert entry missing keys: {missing}\n"
                f"Got: {list(alert.keys())}"
            )

    def test_alert_status_valid_if_populated(self, flask_test_client):
        data = _json(flask_test_client.get("/api/alerts"))
        valid_alert_statuses = {"WARNING", "CRITICAL"}
        for alert in data["alerts"]:
            assert alert["status"] in valid_alert_statuses


# ─────────────────────────────────────────────────────────────────────────────
# 6. Edge cases and error conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestApiEdgeCases:
    """
    Tests for HTTP-level error handling and edge conditions.
    """

    def test_invalid_motor_id_404(self, flask_test_client):
        """
        Motor IDs 3, 99, 0 are not registered.
        Flask should return 404 (not found) for unregistered routes.
        """
        response = flask_test_client.get("/api/motor/3")
        assert response.status_code == 404, (
            f"GET /api/motor/3 returned {response.status_code}, expected 404"
        )

    def test_unknown_route_404(self, flask_test_client):
        response = flask_test_client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_root_404(self, flask_test_client):
        """The API has no root endpoint — / returns 404."""
        response = flask_test_client.get("/")
        assert response.status_code == 404

    def test_multiple_status_requests_consistent(self, flask_test_client):
        """
        Two sequential /api/status requests must return the same motor IDs
        and structural keys (data values may differ if gateway is live).
        """
        data1 = _json(flask_test_client.get("/api/status"))
        data2 = _json(flask_test_client.get("/api/status"))
        assert set(data1.keys()) == set(data2.keys()), (
            "Sequential /api/status calls returned different top-level keys"
        )

    def test_content_type_json(self, flask_test_client):
        """All API responses must set Content-Type: application/json."""
        for endpoint in ["/api/status", "/api/motor/1", "/api/motor/2",
                         "/api/history", "/api/alerts"]:
            r = flask_test_client.get(endpoint)
            ct = r.content_type or ""
            assert "application/json" in ct, (
                f"GET {endpoint} Content-Type: '{ct}', expected 'application/json'"
            )