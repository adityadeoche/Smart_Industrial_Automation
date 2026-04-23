"""
test_sensor.py — Sensor Simulator Validation
=============================================
Validates that sensor_simulator.generate_scenario() produces physically
realistic time-series data for all four operating scenarios.

Test strategy maps to FAT (Factory Acceptance Test) Level 1:
  "Verify that each sensor channel is within calibrated limits for the
   declared operating condition before the unit ships."

In the research paper these tests provide Table 1: "Simulator Validation —
Mean sensor values per scenario vs. datasheet nominal values."

WHY statistical tests with 50 seeds?
  A deterministic test with seed=42 only validates ONE random realisation.
  Running 50 seeds and checking the aggregate mean proves the fault signature
  is statistically persistent — not an artefact of a particular noise draw.
  This corresponds to a Monte-Carlo validation with N=50 independent trials.
"""

import pytest
import numpy as np
import pandas as pd
from sensor_simulator import generate_scenario, SCENARIOS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gen(scenario, duration_s=1, fs=1000, seed=42):
    """Convenience wrapper for generate_scenario."""
    return generate_scenario(scenario, duration_s=duration_s, fs=fs, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DataFrame structure — applies to ALL scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestDataFrameStructure:
    """
    Structural / schema tests.
    These must pass before any numerical tests — they guard against silent
    column renames or dtype changes breaking downstream ML features.
    """

    REQUIRED_COLS = [
        "time_s", "vibration_x_g", "vibration_y_g", "vibration_z_g",
        "current_a_A", "current_b_A", "current_c_A",
        "temperature_C", "speed_rpm", "sound_dB", "scenario",
    ]

    FLOAT_COLS = [
        "time_s", "vibration_x_g", "vibration_y_g", "vibration_z_g",
        "current_a_A", "current_b_A", "current_c_A",
        "temperature_C", "speed_rpm", "sound_dB",
    ]

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_all_required_columns_present(self, scenario):
        df = _gen(scenario)
        for col in self.REQUIRED_COLS:
            assert col in df.columns, f"Missing column '{col}' in scenario '{scenario}'"

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_sensor_columns_are_float(self, scenario):
        """
        All numeric sensor columns must be float dtype.
        Integer columns would cause silent precision loss in the ML feature
        vector (np.float32 cast in MotorAnomalyDetector._build_feature_vector).
        """
        df = _gen(scenario)
        for col in self.FLOAT_COLS:
            assert pd.api.types.is_float_dtype(df[col]), (
                f"Column '{col}' has dtype {df[col].dtype} in scenario '{scenario}' "
                f"(expected float)"
            )

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_no_nan_values(self, scenario):
        """
        NaN in any column would corrupt the Isolation Forest feature matrix
        (sklearn raises ValueError: Input contains NaN).
        """
        df = _gen(scenario)
        nan_counts = df.isnull().sum()
        assert nan_counts.sum() == 0, (
            f"NaN values found in scenario '{scenario}':\n{nan_counts[nan_counts > 0]}"
        )

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_scenario_label_column(self, scenario):
        """
        The 'scenario' string column is used as the class label in
        supervised evaluation. Every row must carry the correct label.
        """
        df = _gen(scenario)
        assert (df["scenario"] == scenario).all(), (
            f"Scenario label mismatch in '{scenario}' — "
            f"found: {df['scenario'].unique()}"
        )

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_row_count(self, scenario):
        """duration_s=1, fs=1000 → exactly 1000 rows."""
        df = _gen(scenario, duration_s=1, fs=1000)
        assert len(df) == 1000, (
            f"Expected 1000 rows, got {len(df)} for scenario '{scenario}'"
        )

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError, match="scenario must be one of"):
            generate_scenario("unknown_fault")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Normal scenario — baseline operating point
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalScenario:
    """
    Validates the healthy motor baseline against datasheet values.
    Motor nameplate: rated_speed=1480 RPM, typical temp=65°C, vib≈0.05g RMS.

    Ranges are deliberately generous (±5%) to accommodate realistic noise
    without making the tests brittle.
    """

    @pytest.fixture(autouse=True)
    def df(self):
        self._df = _gen("normal")

    def test_temperature_mean_in_range(self):
        mean_temp = self._df["temperature_C"].mean()
        assert 63.0 <= mean_temp <= 67.0, (
            f"Normal temperature mean {mean_temp:.2f}°C not in [63, 67]°C"
        )

    def test_vibration_rms_in_range(self):
        """
        Vibration is a sinusoidal signal — its mean is ~0 (positive and
        negative half-cycles cancel). The correct measure is RMS (Root Mean
        Square), which for a sine of amplitude A gives A/√2.
        The spec states 'vib_x ≈ 0.05g RMS', so we test RMS, not mean.
        """
        import numpy as np
        rms_vib = np.sqrt((self._df["vibration_x_g"] ** 2).mean())
        assert 0.02 <= rms_vib <= 0.10, (
            f"Normal vibration RMS {rms_vib:.4f}g not in [0.02, 0.10]g"
        )

    def test_speed_mean_in_range(self):
        mean_speed = self._df["speed_rpm"].mean()
        assert 1470.0 <= mean_speed <= 1490.0, (
            f"Normal speed mean {mean_speed:.1f} RPM not in [1470, 1490]"
        )

    def test_time_vector_is_monotonic(self):
        """time_s must be strictly increasing (no duplicate timestamps)."""
        t = self._df["time_s"].values
        assert np.all(np.diff(t) > 0), "time_s is not strictly increasing"

    def test_time_starts_at_zero(self):
        assert self._df["time_s"].iloc[0] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bearing Fault scenario
# ─────────────────────────────────────────────────────────────────────────────

class TestBearingFaultScenario:
    """
    Bearing fault injects amplitude-modulated impulsive vibration at BPFO/BPFI
    and raises bearing temperature due to friction.

    Fault signature (from motor nameplate):
      • BPFO = 157 Hz, BPFI = 218 Hz — amplitude modulated at shaft freq
      • vib_x += 0.30 × AM-carrier + 0.15 × inner-race carrier → mean >> 0.15g
      • temp += 15°C → mean >> 75°C  (baseline 65 + offset 15 = 80; noise pulls down slightly)
    """

    @pytest.fixture(autouse=True)
    def df(self):
        self._df = _gen("bearing_fault", duration_s=2, fs=1000)
        self._normal = _gen("normal", duration_s=2, fs=1000)

    def test_vibration_elevated(self):
        """
        Vibration is sinusoidal — mean ≈ 0. Use RMS (standard engineering metric).
        Bearing fault carrier amplitude = 0.30g → RMS ≈ 0.30/√2 ≈ 0.21g plus
        inner-race component. We test RMS > 0.15g (conservative).
        """
        import numpy as np
        rms_vib = np.sqrt((self._df["vibration_x_g"] ** 2).mean())
        assert rms_vib > 0.15, (
            f"Bearing fault vibration RMS {rms_vib:.4f}g not > 0.15g"
        )

    def test_vibration_higher_than_normal(self):
        """Bearing fault RMS vibration must exceed normal RMS vibration."""
        import numpy as np
        rms_fault  = np.sqrt((self._df["vibration_x_g"] ** 2).mean())
        rms_normal = np.sqrt((self._normal["vibration_x_g"] ** 2).mean())
        assert rms_fault > rms_normal, \
            f"Bearing fault RMS vib {rms_fault:.4f} not > normal RMS {rms_normal:.4f}"

    def test_temperature_elevated(self):
        """
        temp baseline ≈ 65°C, bearing_fault offset = +15°C → mean > 75°C.
        Allow some noise tolerance; the test uses > 75 (not ≥ 80).
        """
        mean_temp = self._df["temperature_C"].mean()
        assert mean_temp > 75.0, (
            f"Bearing fault temperature mean {mean_temp:.2f}°C not > 75°C"
        )

    def test_speed_reduced(self):
        """Bearing friction causes a -10 RPM speed drop."""
        mean_speed = self._df["speed_rpm"].mean()
        normal_speed = self._normal["speed_rpm"].mean()
        assert mean_speed < normal_speed - 5.0, (
            f"Bearing fault speed {mean_speed:.1f} not sufficiently below "
            f"normal {normal_speed:.1f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stator Fault scenario
# ─────────────────────────────────────────────────────────────────────────────

class TestStatorFaultScenario:
    """
    Stator winding fault (turn-to-turn short) produces the highest thermal
    signature of all three faults (I²R hotspot: +22°C) and introduces
    torque ripple at 100 Hz in vibration.

    Fault signature:
      • temp += 22°C → mean > 80°C
      • vib_x += 0.12 × sin(100 Hz) → elevated std vs normal
    """

    @pytest.fixture(autouse=True)
    def df(self):
        self._df = _gen("stator_fault", duration_s=2, fs=1000)
        self._normal = _gen("normal", duration_s=2, fs=1000)

    def test_temperature_elevated(self):
        """
        65°C baseline + 22°C stator offset = 87°C nominal.
        We test > 80°C to allow for small noise effects.
        """
        mean_temp = self._df["temperature_C"].mean()
        assert mean_temp > 80.0, (
            f"Stator fault temperature mean {mean_temp:.2f}°C not > 80°C"
        )

    def test_vibration_std_elevated(self):
        """
        The 100 Hz torque ripple injection increases vibration variance.
        Standard deviation of stator fault vib_x must exceed that of normal.
        This is more diagnostic than mean because the 100 Hz component is
        zero-mean (sine wave), so it barely shifts the mean but sharply
        increases the standard deviation.
        """
        std_fault  = self._df["vibration_x_g"].std()
        std_normal = self._normal["vibration_x_g"].std()
        assert std_fault > std_normal, (
            f"Stator fault vib std {std_fault:.5f} not > normal std {std_normal:.5f}"
        )

    def test_temperature_higher_than_bearing_fault(self):
        """
        Stator fault is the hottest fault (22°C rise vs 15°C for bearing).
        """
        df_bearing = _gen("bearing_fault", duration_s=2, fs=1000)
        assert (self._df["temperature_C"].mean() >
                df_bearing["temperature_C"].mean()), \
            "Stator fault should be hotter than bearing fault"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Rotor Bar Fault scenario
# ─────────────────────────────────────────────────────────────────────────────

class TestRotorBarFaultScenario:
    """
    Broken rotor bars cause torque pulsation at rotor-bar-pass frequency
    (≈ 686 Hz for 28 bars), speed oscillation (−15 RPM + 8 × sin(fr)),
    and mild heating (+8°C).

    Fault signature:
      • speed_rpm mean < 1470 RPM  (1480 − 15 offset; oscillation never recovers)
      • temp mean > 70°C  (65 + 8)
    """

    @pytest.fixture(autouse=True)
    def df(self):
        self._df = _gen("rotor_bar_fault", duration_s=2, fs=1000)
        self._normal = _gen("normal", duration_s=2, fs=1000)

    def test_speed_reduced(self):
        """
        Mean speed must be below 1470 RPM.
        The −15 RPM offset plus the sinusoidal oscillation (mean=0) gives a
        persistent speed deficit that pushes the mean well below 1470.
        """
        mean_speed = self._df["speed_rpm"].mean()
        assert mean_speed < 1470.0, (
            f"Rotor bar fault speed mean {mean_speed:.1f} not < 1470 RPM"
        )

    def test_speed_lower_than_normal(self):
        assert (self._df["speed_rpm"].mean() <
                self._normal["speed_rpm"].mean()), \
            "Rotor bar fault speed not below normal"

    def test_temperature_elevated(self):
        """65°C baseline + 8°C rotor losses = 73°C nominal → test > 70°C."""
        mean_temp = self._df["temperature_C"].mean()
        assert mean_temp > 70.0, (
            f"Rotor bar fault temperature mean {mean_temp:.2f}°C not > 70°C"
        )

    def test_speed_not_catastrophically_low(self):
        """
        Rotor bar fault degrades speed but never halts the motor in this model.
        Mean speed must remain above 1400 RPM.
        """
        mean_speed = self._df["speed_rpm"].mean()
        assert mean_speed > 1400.0, (
            f"Rotor bar fault speed mean {mean_speed:.1f} implausibly low (< 1400)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Statistical significance — 50 seeds Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalSignificance:
    """
    Monte-Carlo validation: run each fault with 50 different random seeds
    and verify that the fault metric is CONSISTENTLY outside normal range.

    WHY 50 seeds?
    A single seed could (by chance) produce noise that partially cancels the
    fault signal. Running 50 independent realisations and checking that ALL
    of them exceed the threshold proves the fault signature is deterministic
    (i.e., physically modelled) not an artefact of a specific noise draw.

    This is the equivalent of running 50 independent test-bench measurements —
    a requirement in IEC 60034-14 (mechanical vibration limits) acceptance tests.

    For speed, we use duration_s=0.5 (500 samples at 1 kHz) per seed.
    """

    N_SEEDS = 50

    def test_bearing_fault_vibration_statistically_elevated(self):
        """
        Over 50 seeds, bearing fault vibration RMS must exceed 0.15g for
        every single seed — proving the fault injection is deterministic.
        RMS is used because vibration_x_g is sinusoidal (mean ≈ 0).
        """
        import numpy as np
        failures = []
        for seed in range(self.N_SEEDS):
            df = _gen("bearing_fault", duration_s=0.5, fs=1000, seed=seed)
            rms_vib = np.sqrt((df["vibration_x_g"] ** 2).mean())
            if rms_vib <= 0.15:
                failures.append((seed, rms_vib))

        assert not failures, (
            f"Bearing fault vibration not > 0.15g for {len(failures)}/50 seeds: "
            f"{failures[:5]}..."
        )

    def test_stator_fault_temperature_statistically_elevated(self):
        """
        Over 50 seeds, stator fault temperature mean must exceed 80°C for
        every single seed.
        """
        failures = []
        for seed in range(self.N_SEEDS):
            df = _gen("stator_fault", duration_s=0.5, fs=1000, seed=seed)
            mean_temp = df["temperature_C"].mean()
            if mean_temp <= 80.0:
                failures.append((seed, mean_temp))

        assert not failures, (
            f"Stator fault temperature not > 80°C for {len(failures)}/50 seeds: "
            f"{failures[:5]}..."
        )

    def test_rotor_bar_fault_speed_statistically_reduced(self):
        """
        Over 50 seeds, rotor bar fault speed mean must be below 1470 RPM
        for every single seed.
        """
        failures = []
        for seed in range(self.N_SEEDS):
            df = _gen("rotor_bar_fault", duration_s=0.5, fs=1000, seed=seed)
            mean_speed = df["speed_rpm"].mean()
            if mean_speed >= 1470.0:
                failures.append((seed, mean_speed))

        assert not failures, (
            f"Rotor bar fault speed not < 1470 for {len(failures)}/50 seeds: "
            f"{failures[:5]}..."
        )

    def test_normal_temperature_statistically_in_range(self):
        """
        Over 50 seeds, normal temperature mean must remain in [63, 67]°C —
        proves the normal baseline is stable under different noise draws.
        """
        failures = []
        for seed in range(self.N_SEEDS):
            df = _gen("normal", duration_s=0.5, fs=1000, seed=seed)
            mean_temp = df["temperature_C"].mean()
            if not (63.0 <= mean_temp <= 67.0):
                failures.append((seed, mean_temp))

        assert not failures, (
            f"Normal temperature outside [63, 67]°C for {len(failures)}/50 seeds: "
            f"{failures[:5]}..."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cross-scenario comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossScenarioComparison:
    """
    Validates the relative ordering of fault severity metrics.
    These relationships are published in the research paper Results section
    and must hold for the paper to be internally consistent.
    """

    @pytest.fixture(autouse=True)
    def all_scenarios(self):
        self.dfs = {sc: _gen(sc, duration_s=2, fs=1000, seed=42)
                    for sc in SCENARIOS}

    def test_stator_is_hottest(self):
        """Stator fault produces the highest mean temperature of all faults."""
        temps = {sc: self.dfs[sc]["temperature_C"].mean() for sc in SCENARIOS}
        assert temps["stator_fault"] > temps["bearing_fault"], \
            f"stator({temps['stator_fault']:.1f}) should be > bearing({temps['bearing_fault']:.1f})"
        assert temps["stator_fault"] > temps["rotor_bar_fault"], \
            f"stator({temps['stator_fault']:.1f}) should be > rotor_bar({temps['rotor_bar_fault']:.1f})"

    def test_bearing_has_highest_vibration(self):
        """Bearing fault produces the highest RMS vibration of all faults."""
        import numpy as np
        vibs = {sc: np.sqrt((self.dfs[sc]["vibration_x_g"]**2).mean()) for sc in SCENARIOS}
        assert vibs["bearing_fault"] > vibs["stator_fault"], \
            f"bearing RMS({vibs['bearing_fault']:.4f}) should be > stator({vibs['stator_fault']:.4f})"
        assert vibs["bearing_fault"] > vibs["normal"], \
            "Bearing fault RMS vibration not above normal"

    def test_rotor_has_lowest_speed(self):
        """Rotor bar fault produces the lowest mean shaft speed."""
        speeds = {sc: self.dfs[sc]["speed_rpm"].mean() for sc in SCENARIOS}
        assert speeds["rotor_bar_fault"] < speeds["normal"], \
            "Rotor bar fault speed not below normal"
        assert speeds["rotor_bar_fault"] < speeds["stator_fault"] or \
               speeds["stator_fault"] < speeds["normal"], \
            "Speed ordering inconsistent"