#!/usr/bin/env python3
"""Physics-compliance regressions added for the v10.6 audit."""

import numpy as np

import undulator_v11_radia_integrated_v9 as v11


def test_orthogonal_spectral_power_addition():
    f = 2.0e9
    t = np.linspace(0.0, 200.0 / f, 8192, endpoint=False)
    wave = np.cos(2.0 * np.pi * f * t)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])

    field_one = np.column_stack([wave, np.zeros_like(wave), np.zeros_like(wave)])
    field_two = np.column_stack([wave, wave, np.zeros_like(wave)])
    spec_one = v11.get_spec(t, field_one, e1, e2, f)
    spec_two = v11.get_spec(t, field_two, e1, e2, f)

    i1 = int(np.argmax(spec_one["fft"]))
    i2 = int(np.argmax(spec_two["fft"]))
    power_ratio = (spec_two["fft"][i2] / spec_one["fft"][i1]) ** 2
    assert np.isclose(power_ratio, 2.0, rtol=2e-3), power_ratio


def test_compliance_gate_invariants():
    gamma = np.array([10.0, 10.0])
    speed = v11.beta_from_gamma(10.0) * v11.c0
    momentum = gamma[:, None] * v11.me * np.array([[0.0, 0.0, speed], [0.0, 0.0, speed]])
    result = {
        "g_arr": gamma,
        "u": momentum,
        "f0": 1.0e12,
        "f_expected": 1.0e12,
        "energy_accounting": {"relative_mismatch": 1e-4},
        "Stokes": {"P_lin": 0.6, "P_circ": 0.8},
        "t_obs": np.arange(1000, dtype=float) * 1e-14,
        "r_obs": np.array([0.0, 0.0, 100.0]),
        "r": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "lost_to_aperture": False,
        "quantum": {"chi_max": 1e-5},
    }

    class Device:
        lambda_u = 0.05
        metadata = {}

    audit = v11.physics_compliance_assessment(result, Device())
    assert audit["overall_status"] == "PASS", audit
    assert all(row["status"] == "PASS" for row in audit["checks"]), audit

    result["lost_to_aperture"] = True
    failed = v11.physics_compliance_assessment(result, Device())
    assert failed["overall_status"] == "FAIL", failed


if __name__ == "__main__":
    test_orthogonal_spectral_power_addition()
    test_compliance_gate_invariants()
    print("V10.6 PHYSICS COMPLIANCE TEST PASSED")
