"""
conftest.py — Shared pytest fixtures
======================================
WHY FIXTURES instead of setUp/tearDown:
  • Composability: any test file can declare exactly which fixtures it needs
    as function arguments — no inheritance required.
  • Scope control: scope="module" means the fixture is set up ONCE per test
    file, not once per test function. This is critical for MotorAnomalyDetector
    because training an IsolationForest on 200 samples × 100 trees takes ~0.5 s.
    Without module scope, a suite of 20 ML tests would call .train() 20 times.
  • Yield-based cleanup: `yield` cleanly separates setup from teardown within
    a single function — no need to override both setUp and tearDown separately.
  • Parameterisation: fixtures can be parameterised independently of test
    functions, enabling combinatorial testing without duplication.

Industrial analogy:
  These fixtures represent the FAT (Factory Acceptance Test) pre-conditions:
  calibrated sensors, initialised ML models, and a live API gateway — all
  set up once, validated against, then torn down cleanly.
"""

import sys
import os

# ── Ensure project root is on sys.path so imports work from tests/ ────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sensor_simulator import generate_scenario
from ml_gateway import MotorAnomalyDetector, app


# ─────────────────────────────────────────────────────────────────────────────
# Sensor fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def motor1_simulator():
    """
    Generate 1 second of normal operating data at 1 kHz → 1000 rows.
    Represents the Three-Phase Induction Motor (Motor 1) baseline.
    scope="module": created once and shared across all tests in the module
    that request it — avoids regenerating 1000-sample DataFrames repeatedly.
    """
    return generate_scenario("normal", duration_s=1, fs=1000, seed=42)


@pytest.fixture(scope="module")
def motor2_simulator():
    """
    Separate fixture for Motor 2 (centrifugal pump).
    Uses the same underlying simulator because the pump's CAN node
    applies scaling in can_node.py (×0.97 temp, ×1.05 vib) before
    transmission — the simulator itself is motor-agnostic.
    """
    return generate_scenario("normal", duration_s=1, fs=1000, seed=99)


# ─────────────────────────────────────────────────────────────────────────────
# ML model fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_detector1():
    """
    MotorAnomalyDetector for Motor 1, trained on 200 synthetic normal samples.

    WHY module scope?
    IsolationForest with n_estimators=100 takes ~500 ms to train.
    A module-scoped fixture trains once per test file, not once per test
    function — saving several seconds on a full test run.

    WHY n_samples=200?
    200 points in 3-D feature space (temp, vib, rpm) gives the Isolation
    Forest enough density to learn the normal cluster boundary without
    overfitting. Below ~50 samples the forest becomes unstable; above ~500
    there are diminishing returns for this low-dimensional problem.
    """
    d = MotorAnomalyDetector("motor1")
    d.train(n_samples=200)
    return d


@pytest.fixture(scope="module")
def trained_detector2():
    """
    Independent MotorAnomalyDetector for Motor 2 (pump).
    Motor 2 has a different normal envelope (flow_rate instead of RPM,
    slightly cooler temp) so it MUST be a separate model — a shared model
    would learn a blended distribution and produce higher false-positive rates
    for both machines.
    """
    d = MotorAnomalyDetector("motor2")
    d.train(n_samples=200)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Flask API test client fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flask_test_client():
    """
    Flask test client — no live server, no network, fully deterministic.

    WHY Flask test client instead of a real server?
    • No network I/O: requests are dispatched in-process via WSGI
      (Web Server Gateway Interface) — no TCP socket, no port binding.
    • Deterministic: tests cannot fail due to port conflicts or firewall rules.
    • Speed: in-process dispatch is ~100× faster than HTTP over localhost.
    • Isolation: each test run starts with a fresh gateway state because
      the module-level `state` dict is reset when the module is re-imported.

    In TESTING mode Flask disables the error propagation catch so exceptions
    surface as Python tracebacks rather than 500 HTML pages — making test
    failures much easier to diagnose.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client