import numpy as np
import os
import sys
import tempfile
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.signal import find_peaks
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

c0 = 299792458.0
qe = 1.602176634e-19
eps_0 = 8.8541878128e-12
mu_0 = 4 * np.pi * 1e-7
me = 9.1093837015e-31

# ---------------- Semiclassical quantum-radiation correction ----------------
# Set False to recover the purely classical radiation-reaction model.
QUANTUM_CORRECTION = False  # classical default; chi_e is still monitored
QUANTUM_MONITOR = True
RADIATION_REACTION = True
ADVANCED_DIAGNOSTICS = False  # MLE / radial diffusion / Poincare are optional
B_S = 4.414e9  # Schwinger magnetic-field scale [T]
h_planck = 6.62607015e-34  # Planck constant [J s]
hbar_planck = h_planck / (2.0 * np.pi)
eV_J = 1.602176634e-19     # 1 eV in joules

def quantum_chi_from_vB(gamma, v, B):
    """Electron quantum parameter chi_e for E=0.

    chi_e = gamma * |v x B| / (c * B_S), so only the magnetic-field
    component perpendicular to the electron velocity contributes.
    """
    v = np.asarray(v, dtype=float)
    B = np.asarray(B, dtype=float)
    return float(gamma) * np.linalg.norm(np.cross(v, B)) / (c0 * B_S)

def quantum_chi_array(gamma, v, B):
    """Vectorized chi_e for arrays shaped (N, 3)."""
    gamma = np.asarray(gamma, dtype=float)
    v = np.asarray(v, dtype=float)
    B = np.asarray(B, dtype=float)
    return gamma * np.linalg.norm(np.cross(v, B), axis=1) / (c0 * B_S)

def quantum_gaunt_factor(chi):
    """Semiclassical quantum reduction factor for radiation power."""
    chi = np.maximum(np.asarray(chi, dtype=float), 0.0)
    return (
        1.0
        + 4.8 * (1.0 + chi) * np.log1p(1.7 * chi)
        + 2.44 * chi * chi
    ) ** (-2.0 / 3.0)

# ---------------- Ultra-relativistic numerical helpers ----------------
def gamma_from_beta(beta):
    """Stable Lorentz factor for beta=v/c, especially when beta -> 1."""
    beta = float(beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError("beta must satisfy 0 <= beta < 1")
    # (1-beta^2) = (1-beta)(1+beta) avoids one source of cancellation.
    return 1.0 / np.sqrt((1.0 - beta) * (1.0 + beta))

def beta_from_gamma(gamma):
    """Stable beta from gamma without forming 1 - 1/gamma**2 first."""
    gamma = float(gamma)
    if gamma < 1.0:
        raise ValueError("gamma must be >= 1")
    return np.sqrt((gamma - 1.0) * (gamma + 1.0)) / gamma

def ideal_beta_z_from_gamma(gamma, K):
    """Helical-undulator longitudinal beta for the chosen initial condition."""
    gamma = float(gamma)
    a = (1.0 + K * K) / (gamma * gamma)
    if a >= 1.0:
        raise ValueError("gamma is too low for this K: need gamma > sqrt(1+K^2)")
    return np.sqrt(1.0 - a)

def one_minus_ideal_beta_z(gamma, K, beta_z=None):
    """Stable 1-beta_z using (1-beta)(1+beta)=1-beta^2."""
    gamma = float(gamma)
    if beta_z is None:
        beta_z = ideal_beta_z_from_gamma(gamma, K)
    return (1.0 + K * K) / (gamma * gamma * (1.0 + beta_z))

class UndHel:
    """Generalized single-electron insertion-device model with controlled non-idealities.

    Magnetic model:
      * Fundamental + 3rd + 5th harmonics only.
      * Reproducible period-to-period amplitude errors.
      * Reproducible longitudinal period/phase errors.
      * Bx/By transverse imbalance.
      * Bx/By phase mismatch.
      * Optional smooth entrance/exit fringe-field envelope.
      * Optional finite circular aperture for particle-loss diagnostics.

    This remains an analytic engineering field model; a measured 3-D field map
    would be a further realism upgrade.
    """
    handedness = 1

    def K(self, m):
        # Nominal K based on the base peak field.
        return qe * self.B0 * self.lambda_u / (2 * np.pi * m * c0)

    def K_components(self, m):
        K0 = self.K(m)
        return {
            "K0": K0,
            "Kx": abs(self.bx_scale) * K0,
            "Ky": abs(self.by_scale * self.transverse_imbalance) * K0,
            "K_eff_rms": K0 * np.sqrt(
                0.5 * (
                    self.bx_scale ** 2
                    + (self.by_scale * self.transverse_imbalance) ** 2
                )
            ),
        }

    def __init__(
        self, B0, lambda_u, handedness=1,
        h1=0.03, n1=3, h2=0.005, n2=5,
        field_rms=0.0, position_rms=0.0,
        transverse_imbalance=1.0, phase_mismatch=0.0,
        bx_scale=1.0, by_scale=1.0,
        n_error_periods=128, error_seed=12345,
        device_n_periods=100,
        use_fringe_fields=True, fringe_periods=1.0,
        aperture_radius=0.03,
        device_name="helical",
    ):
        self.B0 = float(B0)
        self.lambda_u = float(lambda_u)
        self.k_u = 2 * np.pi / self.lambda_u
        self.handedness = handedness

        # Keep only the physically motivated low odd harmonics used in this study.
        self.h1 = float(h1)
        self.n1 = int(n1)
        self.h2 = float(h2)
        self.n2 = int(n2)
        self.harmonics = {self.n1: self.h1, self.n2: self.h2}

        self.field_rms = max(float(field_rms), 0.0)
        self.position_rms = max(float(position_rms), 0.0)
        self.transverse_imbalance = float(transverse_imbalance)
        self.phase_mismatch = float(phase_mismatch)
        self.bx_scale = float(bx_scale)
        self.by_scale = float(by_scale)
        self.device_name = str(device_name)
        self.n_error_periods = max(int(n_error_periods), 2)
        self.error_seed = int(error_seed)

        self.device_n_periods = max(int(device_n_periods), 1)
        self.device_length = self.device_n_periods * self.lambda_u
        self.use_fringe_fields = bool(use_fringe_fields)
        self.fringe_periods = max(float(fringe_periods), 0.0)
        self.fringe_length = self.fringe_periods * self.lambda_u
        self.aperture_radius = None if aperture_radius is None else max(float(aperture_radius), 0.0)

        rng = np.random.default_rng(self.error_seed)
        self._period_axis = np.arange(-2, self.n_error_periods + 3, dtype=float)
        nerr = len(self._period_axis)
        self._amp_error = rng.normal(0.0, self.field_rms, nerr)
        self._z_error = rng.normal(0.0, self.position_rms, nerr)

    def _local_errors(self, z):
        z_arr = np.asarray(z, dtype=float)
        u = z_arr / self.lambda_u
        amp_err = np.interp(
            u, self._period_axis, self._amp_error,
            left=self._amp_error[0], right=self._amp_error[-1]
        )
        z_err = np.interp(
            u, self._period_axis, self._z_error,
            left=self._z_error[0], right=self._z_error[-1]
        )
        return amp_err, z_err

    @staticmethod
    def _smoothstep01(x):
        """C1 smooth ramp: 0 -> 1 on x in [0,1]."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def envelope(self, z):
        """Smooth finite-length entrance/exit field envelope."""
        z = np.asarray(z, dtype=float)
        if not self.use_fringe_fields or self.fringe_length <= 0.0:
            return np.where((z >= 0.0) & (z <= self.device_length), 1.0, 0.0)

        Lin = self.fringe_length
        L = self.device_length

        env = np.zeros_like(z, dtype=float)

        # Entrance ramp
        m_in = (z >= 0.0) & (z < Lin)
        env[m_in] = self._smoothstep01(z[m_in] / Lin)

        # Flat central region
        m_mid = (z >= Lin) & (z <= L - Lin)
        env[m_mid] = 1.0

        # Exit ramp
        m_out = (z > L - Lin) & (z <= L)
        env[m_out] = self._smoothstep01((L - z[m_out]) / Lin)

        if env.ndim == 0:
            return float(env)
        return env

    def error_summary(self):
        return {
            "field_rms": self.field_rms,
            "position_rms_m": self.position_rms,
            "transverse_imbalance": self.transverse_imbalance,
            "phase_mismatch_rad": self.phase_mismatch,
            "bx_scale": self.bx_scale,
            "by_scale": self.by_scale,
            "device_name": self.device_name,
            "error_seed": self.error_seed,
            "magnetic_harmonics": dict(self.harmonics),
            "magnetic_h3": self.harmonics.get(3, 0.0),
            "magnetic_h5": self.harmonics.get(5, 0.0),
            "device_n_periods": self.device_n_periods,
            "device_length_m": self.device_length,
            "use_fringe_fields": self.use_fringe_fields,
            "fringe_periods": self.fringe_periods,
            "aperture_radius_m": self.aperture_radius,
        }

    def B(self, r):
        r = np.asarray(r)
        scalar = (r.ndim == 1)
        z = np.asarray(r[2] if scalar else r[:, 2], dtype=float)

        amp_err, z_err = self._local_errors(z)
        ph = self.k_u * (z - z_err)
        local_B0 = self.B0 * (1.0 + amp_err) * self.envelope(z)

        sx = np.sin(ph)
        cy = np.cos(ph + self.phase_mismatch)
        for n, h in self.harmonics.items():
            sx = sx + h * np.sin(n * ph)
            cy = cy + h * np.cos(n * ph + self.phase_mismatch)

        bx = local_B0 * self.bx_scale * sx
        by = (
            local_B0
            * self.by_scale
            * self.handedness
            * self.transverse_imbalance
            * cy
        )
        bz = np.zeros_like(z, dtype=float)

        if scalar:
            return np.array([float(bx), float(by), 0.0])
        return np.stack([bx, by, bz], axis=-1)

    def inside_aperture(self, r):
        """Return boolean mask for a circular transverse aperture."""
        if self.aperture_radius is None or self.aperture_radius <= 0.0:
            r = np.asarray(r)
            if r.ndim == 1:
                return True
            return np.ones(len(r), dtype=bool)
        r = np.asarray(r)
        if r.ndim == 1:
            return bool(np.hypot(r[0], r[1]) <= self.aperture_radius)
        return np.hypot(r[:, 0], r[:, 1]) <= self.aperture_radius

def aperture_event(t, state, helical_und, m_part, q_part):
    """Terminate integration when a finite transverse aperture is crossed."""
    ap = helical_und.aperture_radius
    if ap is None or ap <= 0.0:
        return 1.0
    return ap - np.hypot(state[0], state[1])

aperture_event.terminal = True
aperture_event.direction = -1


def field_map_end_event(t, state, device, m_part, q_part):
    """Terminate a real field-map solve exactly at its configured z end."""
    if not getattr(device, "uses_real_end_fields", False):
        return 1.0
    md = dict(getattr(device, "metadata", {}) or {})
    z_end = float(md.get(
        "tracking_z_end_m",
        float(device.z_grid[-1]) if hasattr(device, "z_grid") else np.inf,
    ))
    return z_end - float(state[2])

field_map_end_event.terminal = True
field_map_end_event.direction = -1


def solution_arrays_with_terminal_sample(sol):
    """Return t/y including the exact terminal event state when t_eval missed it."""
    ts = np.asarray(sol.t, dtype=float)
    Y = np.asarray(sol.y, dtype=float)
    if sol.status != 1 or not getattr(sol, "t_events", None):
        return ts, Y

    candidates = []
    y_events = getattr(sol, "y_events", None)
    if y_events is None:
        return ts, Y
    for event_times, event_states in zip(sol.t_events, y_events):
        if len(event_times) and len(event_states):
            candidates.append((float(event_times[-1]), np.asarray(event_states[-1], dtype=float)))
    if not candidates:
        return ts, Y
    te, ye = max(candidates, key=lambda item: item[0])
    tol = 64.0 * np.finfo(float).eps * max(1.0, abs(te))
    if len(ts) == 0 or abs(ts[-1] - te) > tol:
        ts = np.append(ts, te)
        Y = np.column_stack([Y, ye])
    else:
        ts[-1] = te
        Y[:, -1] = ye
    return ts, Y


DEFAULT_INJECTION = {
    # Small deterministic offsets; set all to zero for the original ideal injection.
    "x_offset_m": 0.0,
    "y_offset_m": 0.0,
    "angle_x_rad": 0.0,
    "angle_y_rad": 0.0,
}


def device_K_components(device, m_part=me):
    """Return nominal transverse K amplitudes for the selected device."""
    return device.K_components(m_part)


def device_resonance_factor(device, m_part=me):
    """A = 1 + (Kx^2 + Ky^2)/2 for a generalized transverse periodic device.

    Planar:  A = 1 + K^2/2
    Helical: A = 1 + K^2
    Elliptical: smoothly interpolates between them.
    """
    kc = device_K_components(device, m_part)
    return 1.0 + 0.5 * (kc["Kx"] ** 2 + kc["Ky"] ** 2)


def ideal_beta_z_device(gamma, device, m_part=me):
    """Ultra-relativistic average longitudinal beta for the selected device."""
    A = device_resonance_factor(device, m_part)
    val = 1.0 - A / (float(gamma) ** 2)
    if val <= 0.0:
        return 0.0
    return float(np.sqrt(val))


def fund_lambda_device(gamma, device, theta=0.0, m_part=me):
    """Generalized on/off-axis resonance wavelength.

    lambda_r ≈ lambda_u/(2 gamma^2) *
               [1 + (Kx^2+Ky^2)/2 + gamma^2 theta^2]
    """
    gamma = float(gamma)
    A = device_resonance_factor(device, m_part)
    return device.lambda_u * (A + (gamma * float(theta)) ** 2) / (2.0 * gamma ** 2)


def fund_freq_device(gamma, device, theta=0.0, m_part=me):
    lam = fund_lambda_device(gamma, device, theta=theta, m_part=m_part)
    return c0 / lam if lam > 0.0 else 0.0


def observation_angle_from_vector(r_obs):
    """Total observation angle to the nominal +z device axis."""
    ro = np.asarray(r_obs, dtype=float)
    if ro.shape != (3,):
        raise ValueError("r_obs must be a 3-vector.")
    transverse = float(np.hypot(ro[0], ro[1]))
    longitudinal = abs(float(ro[2]))
    return float(np.arctan2(transverse, longitudinal))


def expected_frequency_from_result(res, r_obs):
    """Generalized off-axis fundamental using the stored device K components."""
    gamma = float(res["gamma_avg"])
    ku = float(res["k_u"])
    lambda_u = 2.0 * np.pi / ku
    kc = dict(res.get("K_components", {}))
    kx = float(kc.get("Kx", 0.0))
    ky = float(kc.get("Ky", 0.0))
    A = 1.0 + 0.5 * (kx*kx + ky*ky)
    theta = observation_angle_from_vector(r_obs)
    lam = lambda_u * (A + (gamma*theta)**2) / (2.0*gamma*gamma)
    return c0/lam if lam > 0.0 else 0.0


def observer_time_bounds(r_obs, r_spl, t0, t1):
    """Causal arrival-time bounds from source-trajectory endpoints."""
    ro = np.asarray(r_obs, dtype=float)
    r0 = np.asarray(r_spl(float(t0)), dtype=float)
    r1 = np.asarray(r_spl(float(t1)), dtype=float)
    to0 = float(t0) + float(np.linalg.norm(ro-r0))/c0
    to1 = float(t1) + float(np.linalg.norm(ro-r1))/c0
    if to1 <= to0:
        raise RuntimeError(
            f"Invalid observer arrival-time interval: {to0} -> {to1}."
        )
    return to0, to1


def observer_signal(
    r_obs, r_spl, v_spl, a_spl, t0, t1, f_exp, *,
    n_obs=None, coarse_samples=1600, min_cycles=10
):
    """Build a fresh observer-specific time window and evaluate LW radiation."""
    to0, to1 = observer_time_bounds(r_obs, r_spl, t0, t1)
    nc = max(400, int(coarse_samples))
    to_coarse = np.linspace(to0, to1, nc)
    Ec, tpc = eval_E(
        to_coarse, r_obs, r_spl, v_spl, a_spl, t0, t1, -qe
    )

    _, e1c, e2c = transv_basis(r_obs)
    p1 = np.dot(Ec, e1c)
    p2 = np.dot(Ec, e2c)
    probe = np.sqrt(
        (p1-np.mean(p1))**2 + (p2-np.mean(p2))**2
    )
    mx = float(np.max(probe)) if len(probe) else 0.0
    valid = probe > 0.001*mx if mx > 0.0 else np.zeros(len(probe), dtype=bool)
    if np.any(valid):
        vi = np.where(valid)[0]
        pad = max(5, int(0.05*(vi[-1]-vi[0]+1)))
        tw0 = float(to_coarse[max(0,vi[0]-pad)])
        tw1 = float(to_coarse[min(nc-1,vi[-1]+pad)])
    else:
        tw0, tw1 = float(to0), float(to1)

    if n_obs is None:
        T = max(tw1-tw0, np.finfo(float).eps)
        n_eval = max(int(64.0*float(f_exp)*T)+1, 8000)
        n_eval = min(n_eval, 80000)
    else:
        n_eval = max(256, int(n_obs))

    to = np.linspace(tw0, tw1, n_eval)
    E, tp = eval_E(to, r_obs, r_spl, v_spl, a_spl, t0, t1, -qe)
    to, E, tp = trim_win(
        to, E, tp, t0, t1, float(f_exp), min_cycles=int(min_cycles)
    )
    return to, E, tp, (tw0, tw1)


def simulation_span_for_device(gamma, device, n_periods=100, t0=0.0, m_part=me):
    beta_z = ideal_beta_z_device(gamma, device, m_part)
    if beta_z <= 0.0:
        raise ValueError("gamma too low for the selected insertion-device K values")

    if getattr(device, "uses_real_end_fields", False):
        md = dict(getattr(device, "metadata", {}) or {})
        z0 = float(md.get(
            "tracking_z_start_m",
            float(device.z_grid[0]) if hasattr(device, "z_grid") else 0.0
        ))
        z1 = float(md.get(
            "tracking_z_end_m",
            float(device.z_grid[-1]) if hasattr(device, "z_grid")
            else z0 + float(n_periods) * device.lambda_u
        ))
        if z1 <= z0:
            raise ValueError(f"Invalid real-field tracking range: z0={z0}, z1={z1}")
        safety = float(md.get("tracking_time_safety_factor", 1.10))
        safety = max(safety, 1.0)
        return float(t0), float(t0) + safety * (z1-z0)/(beta_z*c0)

    return (
        float(t0),
        float(t0) + float(n_periods) * device.lambda_u / (beta_z * c0)
    )


def make_initial_state_device(gamma0, device, injection=None):
    """Device-aware single-electron initial condition.

    The phase convention used by the field model is
        Bx ~ sin(k_u z), By ~ cos(k_u z).
    At z=0 this gives:
      * planar By device: vx=0 at a horizontal turning point;
      * helical device: vx=0 and vy≈Kx*c/gamma;
      * elliptical device: the corresponding intermediate orbit.

    Small user-specified position/angle injection errors are then added.
    """
    inj = dict(DEFAULT_INJECTION)
    if injection is not None:
        inj.update(injection)

    gamma0 = float(gamma0)
    beta = beta_from_gamma(gamma0)
    speed = beta * c0

    # A RADIA/measured finite field has a physical entrance/end field. Start
    # with an electron entering along z rather than imposing the infinite
    # periodic analytic matched orbit.
    if getattr(device, "uses_real_end_fields", False):
        vx = speed * np.tan(float(inj["angle_x_rad"]))
        vy = speed * np.tan(float(inj["angle_y_rad"]))
        vxy2 = vx*vx + vy*vy
        if vxy2 >= speed*speed:
            raise ValueError("Injection angles exceed the total relativistic speed.")
        vz = np.sqrt(speed*speed - vxy2)
        md = dict(getattr(device, "metadata", {}) or {})
        z0 = float(md.get(
            "tracking_z_start_m",
            md.get(
                "tracking_z0_m",
                float(device.z_grid[0]) if hasattr(device, "z_grid") else 0.0
            )
        ))
        init_pos = np.array([
            float(inj["x_offset_m"]),
            float(inj["y_offset_m"]),
            z0,
        ])
        u0 = gamma0 * me * np.array([vx, vy, vz])
        return np.concatenate([init_pos, u0]), {
            "vx_nominal": 0.0,
            "vy_nominal": 0.0,
            "vperp_nominal": 0.0,
            "vz_initial": vz,
            "R_exact_nominal": 0.0,
            "Rx_nominal": 0.0,
            "Ry_nominal": 0.0,
            "injection": inj,
            "device_name": device.device_name,
            "initial_condition_model": "physical_entry",
        }

    kc = device_K_components(device, me)
    Kx, Ky = kc["Kx"], kc["Ky"]

    # Nominal transverse velocity at z=0 from the chosen magnetic phase.
    vy_nom = Kx * c0 / gamma0
    vx_nom = 0.0

    vx_err = speed * np.tan(float(inj["angle_x_rad"]))
    vy_err = speed * np.tan(float(inj["angle_y_rad"]))
    vx = vx_nom + vx_err
    vy = vy_nom + vy_err

    vxy2 = vx * vx + vy * vy
    if vxy2 >= speed * speed:
        raise ValueError("Injection angles/transverse velocity exceed total relativistic speed.")
    vz = np.sqrt(speed * speed - vxy2)

    ku = device.k_u
    # By drives x motion. handedness changes its sign.
    Rx = (
        device.handedness * Ky * c0 / gamma0
        / (ku * max(vz, 1e-30))
    )
    # Bx drives y motion; at z=0 y is near the center for this phase convention.
    Ry = Kx * c0 / gamma0 / (ku * max(vz, 1e-30))

    init_pos = np.array([
        -Rx + float(inj["x_offset_m"]),
        float(inj["y_offset_m"]),
        0.0,
    ])
    u0 = gamma0 * me * np.array([vx, vy, vz])

    return np.concatenate([init_pos, u0]), {
        "vx_nominal": vx_nom,
        "vy_nominal": vy_nom,
        "vperp_nominal": float(np.hypot(vx_nom, vy_nom)),
        "vz_initial": vz,
        "R_exact_nominal": float(np.hypot(Rx, Ry)),
        "Rx_nominal": float(Rx),
        "Ry_nominal": float(Ry),
        "injection": inj,
        "device_name": device.device_name,
    }


# Backward-compatible wrapper for old notebooks.
def make_initial_state(gamma0, K, ku, injection=None):
    """Legacy helical initializer retained for backward compatibility."""
    beta = beta_from_gamma(gamma0)
    speed = beta * c0
    vperp = K * c0 / gamma0
    vx_err = speed * np.tan(float((injection or {}).get("angle_x_rad", 0.0)))
    vy_err = speed * np.tan(float((injection or {}).get("angle_y_rad", 0.0)))
    vx = vx_err
    vy = vperp + vy_err
    vz = np.sqrt(max(speed * speed - vx * vx - vy * vy, 0.0))
    R_ex = vperp / (ku * max(vz, 1e-30))
    inj = dict(DEFAULT_INJECTION)
    if injection:
        inj.update(injection)
    init_pos = np.array([
        -R_ex + float(inj["x_offset_m"]),
        float(inj["y_offset_m"]),
        0.0,
    ])
    u0 = gamma0 * me * np.array([vx, vy, vz])
    return np.concatenate([init_pos, u0]), {
        "vperp_nominal": vperp,
        "vz_initial": vz,
        "R_exact_nominal": R_ex,
        "injection": inj,
    }


def rhs_lorentz(t, state, helical_und, m_part, q_part):
    x = state[0];
    y = state[1];
    z = state[2]
    ux = state[3];
    uy = state[4];
    uz = state[5]
    r = np.array([x, y, z])
    u = np.array([ux, uy, uz])
    u_sq = np.sum(u * u)
    gm = np.power(1 + u_sq / (m_part * m_part * c0 * c0), 0.5)
    v = u / (gm * m_part)
    v_sq = np.sum(v * v)
    B = helical_und.B(r)
    F_lor = q_part * np.cross(v, B)
    if v_sq < 1e-30:
        F_rad = np.zeros(3)
    else:
        cvb = np.cross(v, B)
        numer = (cvb * cvb).sum()
        Pr_classical = mu_0 * q_part ** 4 * gm ** 2 / (6 * np.pi * m_part ** 2 * c0) * numer
        if QUANTUM_CORRECTION:
            chi = quantum_chi_from_vB(gm, v, B)
            Pr = float(quantum_gaunt_factor(chi)) * Pr_classical
        else:
            Pr = Pr_classical
        if RADIATION_REACTION:
            F_rad = -v * (Pr / v_sq)
        else:
            F_rad = np.zeros(3, dtype=float)
    F = F_lor + F_rad
    return [v[0], v[1], v[2], F[0], F[1], F[2]]

def get_a(t_src, r, u, gam, helical_und, m_part, q_part):
    v = u / (gam[:, None] * m_part)
    B = helical_und.B(r)
    Fl = q_part * np.cross(v, B)
    v_sq = np.sum(v * v, axis=1)
    msk = v_sq > 1e-30
    Fr = np.zeros_like(v)
    cvb = np.cross(v[msk], B[msk])
    numer = np.sum(cvb * cvb, axis=1)
    Pr_classical = mu_0 * q_part ** 4 * gam[msk] ** 2 / (6 * np.pi * m_part ** 2 * c0) * numer
    if QUANTUM_CORRECTION:
        chi = quantum_chi_array(gam[msk], v[msk], B[msk])
        Pr = quantum_gaunt_factor(chi) * Pr_classical
    else:
        Pr = Pr_classical
    if RADIATION_REACTION:
        Fr[msk] = -(Pr / v_sq[msk])[:, None] * v[msk]
    else:
        Fr[msk] = 0.0
    du = Fl + Fr
    dg = np.sum(u * du, axis=1) / (gam * m_part * m_part * c0 * c0)
    a = du / (gam[:, None] * m_part)
    a = a - v * (dg / gam)[:, None]
    return a

def quantum_diagnostics(r, u, gam, helical_und, m_part):
    """Return chi_e and Gaunt-factor diagnostics along a solved trajectory."""
    v = u / (gam[:, None] * m_part)
    B = helical_und.B(r)
    chi = quantum_chi_array(gam, v, B)
    gq = quantum_gaunt_factor(chi)
    return {
        "chi_array": chi,
        "g_array": gq,
        "chi_max": float(np.max(chi)),
        "chi_mean": float(np.mean(chi)),
        "g_min": float(np.min(gq)),
        "g_mean": float(np.mean(gq)),
        "quantum_correction_on": bool(QUANTUM_CORRECTION),
    }

def brent_solve(t_obs, r_obs, t_min, t_max, r_spl, v_spl, tol=1e-21, max_iter=120):
    """Causal retarded-time solution using SciPy's bracketed Brent solver."""
    t_obs = float(t_obs)
    a = float(t_min)
    b = float(t_max)
    r_obs_arr = np.asarray(r_obs, dtype=float)

    def residual(t):
        rp = np.asarray(r_spl(float(t)), dtype=float)
        return float(t) + float(np.linalg.norm(r_obs_arr - rp)) / c0 - t_obs

    fa = residual(a)
    fb = residual(b)

    if fa >= 0.0:
        return a
    if fb <= 0.0:
        return b

    return float(
        brentq(
            residual,
            a,
            b,
            xtol=max(float(tol), np.nextafter(0.0, 1.0)),
            rtol=8.0 * np.finfo(float).eps,
            maxiter=int(max_iter),
            disp=True,
        )
    )


def lw_field(tp, r_spl, v_spl, a_spl, r_obs, q_part):
    rp = r_spl(tp)
    vp = v_spl(tp)
    ap = a_spl(tp)
    Rv = r_obs - rp
    R = max(np.power(np.sum(Rv * Rv), 0.5), 1e-12)
    n = Rv / R
    beta = vp / c0
    bdot = ap / c0
    kappa = max(1.0 - np.sum(n * beta), 1e-15)
    cross1 = np.cross(n - beta, bdot)
    numer = np.cross(n, cross1)
    coeff = q_part / (4 * np.pi * eps_0 * c0 * R)
    return coeff * (numer / kappa ** 3), kappa

def transv_basis(r_obs):
    nd = r_obs / np.power(np.sum(r_obs * r_obs), 0.5)
    if abs(nd[0]) < 0.9:
        ref = np.array([1., 0., 0.])
    else:
        ref = np.array([0., 1., 0.])
    e1 = ref - np.sum(ref * nd) * nd
    e1 = e1 / np.power(np.sum(e1 * e1), 0.5)
    e2 = np.cross(nd, e1)
    return nd, e1, e2



def simulation_span_for_gamma(gamma, K, lambda_u, n_periods=100, t0=0.0):
    """Choose source-frame simulation duration from a fixed undulator length.

    Every scan point traverses the same number of undulator periods, so spectra
    at different gamma are compared over the same physical magnetic length.
    """
    beta_z = ideal_beta_z_from_gamma(float(gamma), K)
    vz = beta_z * c0
    t_total = float(n_periods) * lambda_u / vz
    return (float(t0), float(t0) + t_total)

def samples_for_periods(n_periods, pts_per_period=96, min_pts=6000, max_pts=120000):
    """Trajectory samples tied to magnetic periods rather than wall-clock time."""
    return int(np.clip(int(np.ceil(n_periods * pts_per_period)) + 1, min_pts, max_pts))

def n_samples(t_span, vz, lu, pts_per_period=96, min_pts=4000, max_pts=120000):
    dur = t_span[1] - t_span[0]
    n_per = dur * vz / lu
    val = n_per * pts_per_period
    val = np.clip(val, min_pts, max_pts)
    return int(val)

def P_schw(gamma, K, ku, vz=None):
    P0 = mu_0 * qe ** 2 * c0 ** 3 * gamma ** 2 * K ** 2 * ku ** 2 / (6 * np.pi)
    if vz is not None:
        P0 = P0 * (vz / c0) ** 2
        return P0
    return P0


def device_power_K_eff(device, m_part=me):
    """RMS-equivalent K for average radiated-power comparisons.

    Planar sinusoid: K_eff = K/sqrt(2).
    Circular helical: K_eff = K.
    Elliptical devices interpolate through sqrt((Kx^2+Ky^2)/2).
    """
    kc = device.K_components(m_part)
    if "K_eff_rms" in kc:
        return float(kc["K_eff_rms"])
    return float(np.sqrt(0.5 * (kc["Kx"]**2 + kc["Ky"]**2)))


def theoretical_power_device(gamma, device, vz=None, m_part=me):
    return P_schw(
        gamma,
        device_power_K_eff(device, m_part),
        device.k_u,
        vz=vz,
    )

def fund_freq(vz, lu):
    """Observed on-axis fundamental from a supplied longitudinal velocity.

    This remains for simulation-derived vz. For ultra-relativistic theoretical
    predictions, fund_freq_gamma() below is numerically better.
    """
    beta_z = float(vz / c0)
    delta = 1.0 - beta_z
    if delta <= 0.0:
        raise ValueError("vz must be smaller than c")
    return (c0 / lu) * beta_z / delta

def fund_freq_gamma(gamma, K, lu):
    """Stable ideal helical-undulator fundamental for gamma >> 1."""
    beta_z = ideal_beta_z_from_gamma(gamma, K)
    delta = one_minus_ideal_beta_z(gamma, K, beta_z)
    return (c0 / lu) * beta_z / delta

def fund_lambda_gamma(gamma, K, lu):
    """Stable ideal wavelength corresponding to fund_freq_gamma()."""
    beta_z = ideal_beta_z_from_gamma(gamma, K)
    delta = one_minus_ideal_beta_z(gamma, K, beta_z)
    return lu * delta / beta_z

def fit_splines(ts, r, v, a):
    s_r = CubicSpline(ts, r, axis=0)
    s_v = CubicSpline(ts, v, axis=0)
    s_a = CubicSpline(ts, a, axis=0)
    return s_r, s_v, s_a, ts[0], ts[-1]

def eval_E(t_obs, r_obs, r_spl, v_spl, a_spl, t0, t1, q_part):
    n = len(t_obs)
    E = np.zeros((n, 3))
    tp_log = np.zeros(n)
    for i in range(n):
        t = t_obs[i]
        tp = brent_solve(t, r_obs, t0, t1, r_spl, v_spl)
        val, _ = lw_field(tp, r_spl, v_spl, a_spl, r_obs, q_part)
        E[i] = val
        tp_log[i] = tp
    return E, tp_log

def trim_win(t_obs, E, tp_log, t0, t1, fref, min_cycles=30):
    T = 1.0 / max(fref, 1e6)
    margin = 5.0 * T
    ok = (tp_log > t0 + margin) & (tp_log < t1 - margin)
    if np.count_nonzero(ok) < min_cycles:
        return t_obs, E, tp_log
    return t_obs[ok], E[ok], tp_log[ok]

def parab_peak(freq, amp, ip):
    if ip <= 0:
        return float(freq[ip])
    if ip >= len(amp) - 1:
        return float(freq[ip])
    y0 = amp[ip - 1]
    y1 = amp[ip]
    y2 = amp[ip + 1]
    den = y0 - 2.0 * y1 + y2
    if abs(den) < 1e-30 * max(y1, 1e-30):
        return float(freq[ip])
    delta = 0.5 * (y0 - y2) / den
    df = freq[ip + 1] - freq[ip]
    return float(freq[ip] + delta * df)

def phasor(t, sig, f_hz, window):
    t0 = t - t[0]
    ws = window.sum()
    arg = -2j * np.pi * f_hz * t0
    factor = np.exp(arg)
    tmp = sig * window * factor
    return tmp.sum() * (2.0 / ws)


def trajectory_phase_diagnostics(ts, r, u, device, m_part=me):
    """Quantify orbit repeatability, phase slippage, and exit steering.

    Samples the trajectory once per undulator period using z as the phase
    coordinate. These diagnostics are more directly relevant to insertion-
    device quality than the legacy chaos metrics.
    """
    ts = np.asarray(ts, dtype=float)
    r = np.asarray(r, dtype=float)
    u = np.asarray(u, dtype=float)
    if len(ts) < 3:
        return {}

    gam = np.sqrt(1.0 + np.sum(u*u, axis=1)/(m_part*m_part*c0*c0))
    v = u / (gam[:, None] * m_part)

    z = r[:, 2]
    z0 = z[0]
    z1 = z[-1]
    if z1 <= z0:
        return {}

    n_periods = int(np.floor((z1 - z0) / device.lambda_u))
    if n_periods < 1:
        return {}

    sample_z = z0 + np.arange(n_periods + 1) * device.lambda_u
    x_s = np.interp(sample_z, z, r[:, 0])
    y_s = np.interp(sample_z, z, r[:, 1])
    vx_s = np.interp(sample_z, z, v[:, 0])
    vy_s = np.interp(sample_z, z, v[:, 1])
    vz_s = np.interp(sample_z, z, v[:, 2])

    # Period-to-period repeatability error in position.
    dx = np.diff(x_s)
    dy = np.diff(y_s)
    repeat_rms = float(np.sqrt(np.mean(dx*dx + dy*dy))) if len(dx) else 0.0
    repeat_max = float(np.max(np.sqrt(dx*dx + dy*dy))) if len(dx) else 0.0

    # Exit steering relative to local longitudinal velocity.
    vz_end = max(abs(vz_s[-1]), 1e-30)
    xprime_exit = float(vx_s[-1] / vz_end)
    yprime_exit = float(vy_s[-1] / vz_end)

    # Geometric phase progression of the transverse orbit.
    orbit_phase = np.unwrap(np.arctan2(y_s, x_s))
    if len(orbit_phase) >= 2:
        ideal_step = np.median(np.diff(orbit_phase))
        phase_step_err = np.diff(orbit_phase) - ideal_step
        phase_step_rms = float(np.sqrt(np.mean(phase_step_err**2)))
        phase_step_max = float(np.max(np.abs(phase_step_err)))
    else:
        ideal_step = 0.0
        phase_step_rms = 0.0
        phase_step_max = 0.0

    r_perp = np.hypot(r[:, 0], r[:, 1])

    return {
        "n_sampled_periods": n_periods,
        "max_transverse_excursion_m": float(np.max(r_perp)),
        "rms_transverse_excursion_m": float(np.sqrt(np.mean(r_perp*r_perp))),
        "period_repeatability_rms_m": repeat_rms,
        "period_repeatability_max_m": repeat_max,
        "exit_xprime_rad": xprime_exit,
        "exit_yprime_rad": yprime_exit,
        "orbit_phase_step_rad": float(ideal_step),
        "orbit_phase_error_rms_rad": phase_step_rms,
        "orbit_phase_error_max_rad": phase_step_max,
    }


def theory_residuals(sim_frequency_hz, sim_power_W, gamma_avg, device, theta=0.0):
    """Return theory-vs-simulation residuals for the core observables."""
    f_th = fund_freq_device(gamma_avg, device, theta=theta)
    freq_resid = (
        (float(sim_frequency_hz) - f_th) / f_th if f_th > 0 else np.nan
    )

    # Use the existing ideal analytical power helper if available later;
    # power residual is filled by run_sim once its ideal power is known.
    return {
        "frequency_theory_hz": float(f_th),
        "frequency_relative_residual": float(freq_resid),
    }


def spectral_fwhm(freq, amplitude, peak_frequency):
    """Estimate spectral FWHM around the selected peak using linear interpolation."""
    freq = np.asarray(freq, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    if len(freq) < 3 or peak_frequency <= 0.0 or not np.any(np.isfinite(amp)):
        return 0.0, 0.0, 0.0

    # Use spectral power for the half-maximum definition.
    power = np.maximum(amp, 0.0) ** 2
    ip = int(np.argmin(np.abs(freq - peak_frequency)))
    pmax = power[ip]
    if pmax <= 0.0:
        return 0.0, 0.0, 0.0
    half = 0.5 * pmax

    il = ip
    while il > 0 and power[il] >= half:
        il -= 1
    ir = ip
    while ir < len(power) - 1 and power[ir] >= half:
        ir += 1

    if il == ip or ir == ip:
        return 0.0, 0.0, 0.0

    def cross_x(i0, i1):
        x0, x1 = freq[i0], freq[i1]
        y0, y1 = power[i0], power[i1]
        if abs(y1 - y0) < 1e-300:
            return 0.5 * (x0 + x1)
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    f_left = cross_x(il, il + 1)
    f_right = cross_x(ir - 1, ir)
    width = max(float(f_right - f_left), 0.0)
    rel = width / peak_frequency if peak_frequency > 0 else 0.0
    Q = peak_frequency / width if width > 0 else 0.0
    return width, rel, Q



def photon_energy_from_frequency(freq_hz):
    """Photon energy in J/eV/keV from frequency."""
    freq_hz = float(max(freq_hz, 0.0))
    E_J = h_planck * freq_hz
    E_eV = E_J / eV_J
    return {
        "J": E_J,
        "eV": E_eV,
        "keV": E_eV / 1e3,
    }


def harmonic_ratios(freq, spectral_amp, f0, harmonics=(3, 5), half_width_f0=0.25):
    """Integrated radiation-harmonic power ratios.

    This integrates |FFT|^2 in a finite band around n*f0 instead of comparing
    a single peak/bin, making the diagnostic less sensitive to FFT resolution.
    These are RADIATION harmonic ratios, not magnetic-field harmonic amplitudes.
    """
    freq = np.asarray(freq, dtype=float)
    amp = np.asarray(spectral_amp, dtype=float)
    power = np.maximum(amp, 0.0) ** 2
    out = {}
    if len(freq) < 3 or f0 <= 0.0:
        for n in harmonics:
            out[f"H{n}_over_H1"] = np.nan
        return out

    def band_integral(center):
        half = float(half_width_f0) * f0
        msk = (freq >= center - half) & (freq <= center + half)
        if np.count_nonzero(msk) < 2:
            return 0.0
        return float(np.trapezoid(power[msk], freq[msk]))

    p1 = band_integral(f0)
    for n in harmonics:
        pn = band_integral(n * f0)
        out[f"H{n}_over_H1"] = pn / p1 if p1 > 0.0 else np.nan
    return out



def spectral_photon_yield_estimate(
    freq,
    spectral_amp,
    total_radiated_energy_J,
    duration_s=None,
):
    """Estimate photon count using the simulated spectral shape.

    The positive-frequency |FFT|^2 shape is normalized so its integral equals
    the independently computed total radiated energy. Photon number is then

        N_gamma = ∫ [dE/df] / (h f) df.

    Important limitation: the FFT is for the selected observer direction, while
    total_radiated_energy_J is an all-direction power estimate. Therefore this
    is a useful spectral-shape-weighted estimate, not a full 4π photon-flux
    calculation.
    """
    f = np.asarray(freq, dtype=float)
    amp = np.asarray(spectral_amp, dtype=float)
    p = np.maximum(amp, 0.0) ** 2
    valid = (f > 0.0) & np.isfinite(f) & np.isfinite(p)
    f = f[valid]
    p = p[valid]
    if len(f) < 2 or total_radiated_energy_J <= 0.0:
        return {
            "spectral_photons_estimate": 0.0,
            "spectral_photon_rate_estimate_s^-1": 0.0,
        }

    norm = float(np.trapezoid(p, f))
    if norm <= 0.0:
        return {
            "spectral_photons_estimate": 0.0,
            "spectral_photon_rate_estimate_s^-1": 0.0,
        }

    dE_df = float(total_radiated_energy_J) * p / norm
    integrand = dE_df / (h_planck * f)
    N = float(np.trapezoid(integrand, f))
    rate = N / duration_s if duration_s is not None and duration_s > 0.0 else 0.0
    return {
        "spectral_photons_estimate": N,
        "spectral_photon_rate_estimate_s^-1": float(rate),
    }


def approximate_photon_yield(E_radiated_J, representative_frequency_hz, duration_s=None):
    """Fundamental-equivalent photon yield and rate.

    This divides total radiated energy by h*f_rep. Because the real spectrum
    is broadband/harmonic-rich, this is explicitly an approximate
    'fundamental-equivalent' photon count, not an exact spectrally integrated
    photon number.
    """
    E_ph = h_planck * max(float(representative_frequency_hz), 0.0)
    if E_ph <= 0:
        return {
            "equivalent_photons": 0.0,
            "equivalent_photon_rate_s^-1": 0.0,
        }
    N = max(float(E_radiated_J), 0.0) / E_ph
    rate = N / duration_s if duration_s is not None and duration_s > 0 else 0.0
    return {
        "equivalent_photons": float(N),
        "equivalent_photon_rate_s^-1": float(rate),
    }

def get_spec(t_obs, Eo, e1, e2, f_exp):
    E1 = np.dot(Eo, e1)
    E2 = np.dot(Eo, e2)
    n = len(t_obs)
    dt = t_obs[1] - t_obs[0]
    w = np.hanning(n)
    d1 = E1 - E1.mean()
    d2 = E2 - E2.mean()
    s1 = np.fft.fft(d1 * w)
    s2 = np.fft.fft(d2 * w)
    freq = np.fft.fftfreq(n, dt)
    pos = freq > 0
    fp = freq[pos]
    amp = np.abs(s1[pos]) + np.abs(s2[pos])
    nyq = 0.95 * 0.5 / dt
    keep = fp < nyq
    fp = fp[keep]
    amp = amp[keep]
    f0 = 0.0
    if len(amp) > 0:
        band = (fp > 0.7 * f_exp) & (fp < 1.3 * f_exp)
        if np.any(band):
            il = int(np.argmax(amp[band]))
            ip = int(np.where(band)[0][il])
        else:
            ip = int(np.argmax(amp))
        f0 = parab_peak(fp, amp, ip)
    c1 = phasor(t_obs, E1, f0, w)
    c2 = phasor(t_obs, E2, f0, w)
    I = abs(c1) ** 2 + abs(c2) ** 2
    Q = abs(c1) ** 2 - abs(c2) ** 2
    U = 2.0 * np.real(c1 * np.conj(c2))
    V = 2.0 * np.imag(c1 * np.conj(c2))
    if I > 1e-30:
        P_lin = min(np.power(Q ** 2 + U ** 2, 0.5) / I, 1.0)
        P_circ = float(np.clip(V / I, -1.0, 1.0))
    else:
        P_lin = 0.0
        P_circ = 0.0
    fwhm_hz, rel_linewidth, quality_factor = spectral_fwhm(fp, amp, f0)
    photon_energy = photon_energy_from_frequency(f0)
    harmonic_diag = harmonic_ratios(fp, amp, f0, harmonics=(3, 5))
    return {
        "E1": E1, "E2": E2, "f0": f0, "fp": fp, "fft": amp,
        "fwhm_hz": fwhm_hz,
        "relative_linewidth": rel_linewidth,
        "quality_factor": quality_factor,
        "photon_energy": photon_energy,
        "harmonic_ratios": harmonic_diag,
        "Stokes": {
            "I": I, "Q": Q, "U": U, "V": V,
            "P_lin": P_lin, "P_circ": P_circ
        }
    }

def steady_vals(ts, u, m_part, steady_frac=(0.15, 0.85)):
    n = len(ts)
    i0 = int(steady_frac[0] * n)
    i1 = int(steady_frac[1] * n)
    us = u[i0:i1]
    tmp = np.sum(us * us, axis=1)
    gs = np.power(1.0 + tmp / (m_part * m_part * c0 * c0), 0.5)
    vs = us / (gs[:, None] * m_part)
    gz = float(gs.mean())
    vzz = float(vs[:, 2].mean())
    return gz, vzz

def orbit_data(t_arr, r, helical_und, vz):
    x = r[:, 0] - r[:, 0].mean()
    y = r[:, 1] - r[:, 1].mean()
    rperp = np.sqrt(x * x + y * y)
    dt = np.median(np.diff(t_arr))
    pdist = max(int(0.4 * helical_und.lambda_u / (vz * dt)), 3)
    peaks, _ = find_peaks(x, distance=pdist)
    if len(peaks) >= 2:
        pitch = np.mean(np.diff(r[:, 2][peaks]))
    else:
        pitch = helical_und.lambda_u
    Ax = (np.max(x) - np.min(x)) / 2.0
    Ay = (np.max(y) - np.min(y)) / 2.0
    mx = max(Ax, Ay)
    if mx > 1e-20:
        circ = min(Ax, Ay) / mx
    else:
        circ = 1.0
    return {"avg_radius": float(rperp.mean()), "max_radius": float(rperp.max()),
            "pitch": float(pitch), "circularity": float(circ)}

def pulse_data(t_arr, E_sig, fc):
    if len(E_sig) == 0:
        return {"peak_amplitude": 0.0, "avg_fwhm": 0.0, "repetition_freq": 0.0}
    Ec = E_sig - np.mean(E_sig)
    dt = t_arr[1] - t_arr[0]
    pdist = max(int(0.35 / max(fc, 1e6) / dt), 3)
    peaks, _ = find_peaks(np.abs(Ec), distance=pdist, height=np.max(np.abs(Ec)) * 0.05)
    if len(peaks) == 0:
        pa = float(np.max(np.abs(Ec)))
        return {"peak_amplitude": pa, "avg_fwhm": 0.0, "repetition_freq": float(fc)}
    pa = float(np.mean(np.abs(Ec[peaks])))
    hm = pa / 2.0
    fwhms = []
    for p in peaks:
        left = p
        right = p
        while left > 0 and abs(Ec[left]) > hm:
            left = left - 1
        while right < len(Ec) - 1 and abs(Ec[right]) > hm:
            right = right + 1
        if right > left:
            fwhms.append(t_arr[right] - t_arr[left])
    rep = fc
    if len(peaks) >= 2:
        t_diffs = np.diff(t_arr[peaks])
        rep = 1.0 / np.mean(t_diffs) / 2.0
    return {"peak_amplitude": pa, "avg_fwhm": float(np.mean(fwhms)) if fwhms else 0.0,
            "repetition_freq": float(rep)}

def instant_P(r, u, helical_und, m_part, q_part, quantum_corrected=True):
    """Instantaneous radiated power used for energy accounting.

    When quantum_corrected=True and QUANTUM_CORRECTION is enabled, this uses
    the same Gaunt-factor correction as the radiation-reaction force.
    """
    tmp = np.sum(u * u, axis=1)
    gs = np.sqrt(1.0 + tmp / (m_part * m_part * c0 * c0))
    v = u / (gs[:, None] * m_part)
    B = helical_und.B(r)
    cvb = np.cross(v, B)
    P_classical = (
        mu_0 * q_part ** 4 * gs ** 2 / (6 * np.pi * m_part ** 2 * c0)
    ) * np.sum(cvb * cvb, axis=1)

    if quantum_corrected and QUANTUM_CORRECTION:
        chi = quantum_chi_array(gs, v, B)
        return quantum_gaunt_factor(chi) * P_classical
    return P_classical


def energy_accounting(ts, gam, P_rad, m_part=me):
    """Check radiative energy loss against the integrated radiation power."""
    E0 = float(gam[0] * m_part * c0 * c0)
    E1 = float(gam[-1] * m_part * c0 * c0)
    dE_particle = E0 - E1
    E_radiated = float(np.trapezoid(P_rad, ts))
    scale = max(abs(dE_particle), abs(E_radiated), 1e-40)
    residual = dE_particle - E_radiated
    mismatch = residual / scale
    mismatch_initial = residual / max(abs(E0), 1e-40)
    return {
        "E_initial_J": E0,
        "E_final_J": E1,
        "particle_energy_loss_J": dE_particle,
        "integrated_radiated_energy_J": E_radiated,
        "balance_residual_J": float(residual),
        "relative_mismatch": float(mismatch),
        "mismatch_fraction_initial_energy": float(mismatch_initial),
    }

def intensity(r_spl, v_spl, a_spl, t_emit_range, q_part, z_obs=100.0, grid_n=100, angle_max=0.01):
    t_lo = t_emit_range[0]
    t_hi = t_emit_range[1]
    tp = (t_lo + t_hi) / 2.0
    thx = np.linspace(-angle_max, angle_max, grid_n)
    thy = np.linspace(-angle_max, angle_max, grid_n)
    TX, TY = np.meshgrid(thx, thy)
    X = TX * z_obs
    Y = TY * z_obs
    Z = np.full_like(X, z_obs)
    Rg = np.stack([X, Y, Z], axis=-1)
    rp = r_spl(tp)
    vp = v_spl(tp)
    ap = a_spl(tp)
    Rv = Rg - rp
    Rm = np.power(np.sum(Rv * Rv, axis=-1), 0.5)
    nv = Rv / Rm[..., None]
    beta = vp / c0
    bdot = ap / c0
    kappa = 1.0 - np.sum(nv * beta, axis=-1)
    c1 = np.cross(nv - beta, bdot)
    numer = np.cross(nv, c1)
    coeff = q_part / (4 * np.pi * eps_0 * c0 * Rm)
    Em = coeff[..., None] * (numer / kappa[..., None] ** 3)
    return TX, TY, np.sum(Em * Em, axis=-1)

def run_sim(helical_und, v0, t_span, r_obs, n_base=5000, gamma0_input=None, injection=None, rtol=1e-9, atol=1e-11):
    if gamma0_input is None:
        beta0 = v0 / c0
        gamma0 = gamma_from_beta(beta0)
    else:
        gamma0 = float(gamma0_input)
        beta0 = beta_from_gamma(gamma0)
        v0 = beta0 * c0
    K = helical_und.K(me)
    ku = helical_und.k_u
    if ideal_beta_z_device(gamma0, helical_und) <= 0.0:
        raise ValueError("gamma too low for selected device and K components")
    state0, init_meta = make_initial_state_device(gamma0, helical_und, injection=injection)
    vperp = init_meta["vperp_nominal"]
    vz = init_meta["vz_initial"]
    R_ex = init_meta["R_exact_nominal"]

    n_coarse = max(256, int(n_base))
    t_eval = np.linspace(t_span[0], t_span[1], n_coarse)

    sol = solve_ivp(
        rhs_lorentz, t_span, state0,
        args=(helical_und, me, -qe),
        t_eval=t_eval, method='RK45',
        rtol=rtol, atol=atol,
        events=(aperture_event, field_map_end_event)
    )
    if not sol.success:
        raise RuntimeError("ODE failed: " + str(sol.message))

    ts, sol_y = solution_arrays_with_terminal_sample(sol)
    r = sol_y[:3].T
    lost_to_aperture = bool(sol.t_events and len(sol.t_events[0]) > 0)
    reached_field_map_end = bool(
        getattr(helical_und, "uses_real_end_fields", False)
        and len(sol.t_events) > 1 and len(sol.t_events[1]) > 0
    )
    if getattr(helical_und, "uses_real_end_fields", False) and not lost_to_aperture and not reached_field_map_end:
        raise RuntimeError("Real field-map integration ended before the exact configured z_end event.")
    loss_time = float(sol.t_events[0][0]) if lost_to_aperture else None
    u = sol_y[3:].T
    tmp = np.sum(u * u, axis=1)
    ga = np.power(1.0 + tmp / (me * me * c0 * c0), 0.5)
    v = u / (ga[:, None] * me)
    a = get_a(ts, r, u, ga, helical_und, me, -qe)
    qdiag = quantum_diagnostics(r, u, ga, helical_und, me)
    traj_phase = trajectory_phase_diagnostics(ts, r, u, helical_und, me)

    g_avg, vz_avg = steady_vals(ts, u, me)
    theta_obs = observation_angle_from_vector(r_obs)
    # Device-generalized on/off-axis resonance.
    f_exp = fund_freq_device(g_avg, helical_und, theta=theta_obs)
    lam_th = fund_lambda_device(g_avg, helical_und, theta=theta_obs)

    r_spl, v_spl, a_spl, t0, t1 = fit_splines(ts, r, v, a)
    t_obs2, E2, tp_log, (tw0, tw1) = observer_signal(
        r_obs, r_spl, v_spl, a_spl, t0, t1, f_exp,
        n_obs=None, coarse_samples=min(max(n_coarse,800),4000),
        min_cycles=30,
    )

    _, e1, e2 = transv_basis(r_obs)
    spec = get_spec(t_obs2, E2, e1, e2, f_exp)
    stokes = spec["Stokes"]
    f0 = spec["f0"]
    lam0 = c0 / f0 if f0 > 1e-9 else 0.0

    traj = orbit_data(ts, r, helical_und, vz_avg)
    E1 = spec["E1"]
    Emag2 = np.linalg.norm(E2, axis=1)
    if f0 > 0:
        pulse = pulse_data(t_obs2, E1, f0)
    else:
        pulse = pulse_data(t_obs2, E1, f_exp)

    Plt = instant_P(r, u, helical_und, me, -qe)
    n_src = len(ts)
    i_lo = int(0.1 * n_src)
    i_hi = max(int(0.9 * n_src), i_lo + 1)
    Pl = float(np.mean(Plt[i_lo:i_hi]))
    energy_check = energy_accounting(ts, ga, Plt, me)
    photon_yield = approximate_photon_yield(
        energy_check['integrated_radiated_energy_J'],
        spec['f0'],
        duration_s=(ts[-1] - ts[0])
    )
    spectral_photon_yield = spectral_photon_yield_estimate(
        spec["fp"],
        spec["fft"],
        energy_check["integrated_radiated_energy_J"],
        duration_s=(ts[-1] - ts[0]),
    )
    residuals = theory_residuals(spec["f0"], Pl, g_avg, helical_und, theta=theta_obs)

    K_power_eff = device_power_K_eff(helical_und, me)
    Ps_init = theoretical_power_device(gamma0, helical_und, vz=vz, m_part=me)
    Ps_avg = theoretical_power_device(g_avg, helical_und, vz=vz_avg, m_part=me)

    return {
        "gamma0": gamma0, "gamma_avg": g_avg, "K": K,
        "K_power_eff_rms": K_power_eff,
        "v_perp": vperp, "v_z": vz, "v_z_avg": vz_avg, "k_u": ku,
        "r": r, "u": u, "t_src": ts, "g_arr": ga,
        "P_larmor_t": Plt,
        "t_obs": t_obs2, "E": E2, "E_total": Emag2,
        "E1": E1, "E2": spec["E2"],
        "traj": traj, "pulse": pulse,
        "Stokes": stokes,
        "f0": f0, "lambda0": lam0,
        "f_expected": f_exp,
        "f_expected_init": fund_freq_device(gamma0, helical_und, theta=theta_obs),
        "lam_theory": lam_th,
        "P_schwinger": Ps_avg, "P_schwinger_init": Ps_init,
        "P_larmor": Pl, "R_exact": R_ex,
        "freq": spec["fp"], "fft": spec["fft"],
        "spectral_fwhm_hz": spec["fwhm_hz"],
        "relative_linewidth": spec["relative_linewidth"],
        "spectral_quality_factor": spec["quality_factor"],
        "photon_energy": spec["photon_energy"],
        "harmonic_ratios": spec["harmonic_ratios"],
        "photon_yield": photon_yield,
        "spectral_photon_yield": spectral_photon_yield,
        "wiggler_critical_energy": wiggler_critical_energy(helical_und, g_avg),
        "trajectory_phase": traj_phase,
        "theory_residuals": residuals,
        "energy_accounting": energy_check,
        "lost_to_aperture": lost_to_aperture,
        "reached_field_map_end": reached_field_map_end,
        "loss_time_s": loss_time,
        "injection": init_meta["injection"],
        "device_name": helical_und.device_name,
        "K_components": helical_und.K_components(me),
        "splines": (r_spl, v_spl, a_spl, t0, t1),
        "tp_log": tp_log,
        "tw": (tw0, tw1),
        "quantum": qdiag,
        "observer_theta_rad": theta_obs,
        "r_obs": np.asarray(r_obs,dtype=float).copy(),
        "n_source_samples": int(len(ts)),
    }

def run_sim_scalar(und, v0, t_span, r_obs, n_base=4000, gamma0_input=None, injection=None, rtol=1e-9, atol=1e-11):
    if gamma0_input is None:
        beta0 = v0 / c0
        gamma0 = gamma_from_beta(beta0)
    else:
        gamma0 = float(gamma0_input)
        beta0 = beta_from_gamma(gamma0)
        v0 = beta0 * c0
    K = und.K(me)
    ku = und.k_u
    if ideal_beta_z_device(gamma0, und) <= 0.0:
        return None
    state0, init_meta = make_initial_state_device(gamma0, und, injection=injection)
    vperp = init_meta["vperp_nominal"]
    vz = init_meta["vz_initial"]
    R_ex = init_meta["R_exact_nominal"]

    n_coarse = max(256, int(n_base))
    t_eval = np.linspace(t_span[0], t_span[1], n_coarse)

    sol = solve_ivp(
        rhs_lorentz, t_span, state0,
        args=(und, me, -qe),
        t_eval=t_eval, method='RK45',
        rtol=rtol, atol=atol,
        events=(aperture_event, field_map_end_event)
    )
    if not sol.success:
        return None

    ts, sol_y = solution_arrays_with_terminal_sample(sol)
    r = sol_y[:3].T
    lost_to_aperture = bool(sol.t_events and len(sol.t_events[0]) > 0)
    reached_field_map_end = bool(
        getattr(und, "uses_real_end_fields", False)
        and len(sol.t_events) > 1 and len(sol.t_events[1]) > 0
    )
    if getattr(und, "uses_real_end_fields", False) and not lost_to_aperture and not reached_field_map_end:
        raise RuntimeError("Real field-map integration ended before the exact configured z_end event.")
    loss_time = float(sol.t_events[0][0]) if lost_to_aperture else None
    u = sol_y[3:].T
    tmp = np.sum(u * u, axis=1)
    ga = np.power(1.0 + tmp / (me * me * c0 * c0), 0.5)
    v = u / (ga[:, None] * me)
    a = get_a(ts, r, u, ga, und, me, -qe)
    qdiag = quantum_diagnostics(r, u, ga, und, me)
    traj_phase = trajectory_phase_diagnostics(ts, r, u, und, me)

    g_avg, vz_avg = steady_vals(ts, u, me)
    theta_obs = observation_angle_from_vector(r_obs)
    f_exp = fund_freq_device(g_avg, und, theta=theta_obs)
    lam_th = fund_lambda_device(g_avg, und, theta=theta_obs)

    r_spl, v_spl, a_spl, t0, t1 = fit_splines(ts, r, v, a)
    t_obs2, E2, tp_log, (tw0, tw1) = observer_signal(
        r_obs, r_spl, v_spl, a_spl, t0, t1, f_exp,
        n_obs=None, coarse_samples=min(max(n_coarse,800),4000),
        min_cycles=30,
    )

    _, e1, e2 = transv_basis(r_obs)
    spec = get_spec(t_obs2, E2, e1, e2, f_exp)
    stokes = spec["Stokes"]
    f0 = spec["f0"]
    lam0 = c0 / f0 if f0 > 1e-9 else 0.0

    traj = orbit_data(ts, r, und, vz_avg)
    E1 = spec["E1"]
    pulse = pulse_data(t_obs2, E1, f0 if f0 > 0 else f_exp)

    Plt = instant_P(r, u, und, me, -qe)
    n_src = len(ts)
    i_lo, i_hi = int(0.1 * n_src), max(int(0.9 * n_src), int(0.1 * n_src) + 1)
    Pl = float(np.mean(Plt[i_lo:i_hi]))
    energy_check = energy_accounting(ts, ga, Plt, me)
    photon_yield = approximate_photon_yield(
        energy_check['integrated_radiated_energy_J'],
        spec['f0'],
        duration_s=(ts[-1] - ts[0])
    )
    spectral_photon_yield = spectral_photon_yield_estimate(
        spec["fp"],
        spec["fft"],
        energy_check["integrated_radiated_energy_J"],
        duration_s=(ts[-1] - ts[0]),
    )
    residuals = theory_residuals(spec["f0"], Pl, g_avg, und, theta=theta_obs)
    K_power_eff = device_power_K_eff(und, me)
    Ps_avg = theoretical_power_device(g_avg, und, vz=vz_avg, m_part=me)

    res_min = {
        'r': r, 'u': u, 'g_arr': ga,
        'traj': traj, 'v_z_avg': vz_avg,
        'splines': (r_spl, v_spl, a_spl, t0, t1),
        'tw': (tw0, tw1),
        'quantum': qdiag
    }
    if ADVANCED_DIAGNOSTICS:
        ch = chaos_analysis(res_min, und)
    else:
        ch = {
            "mle": np.nan,
            "mle_count": 0,
            "r_diffusion": np.nan,
            "n_poincare": 0,
        }

    return {
        "v0": v0,
        "gamma0": gamma0,
        "gamma_avg": g_avg,
        "v_z": vz,
        "v_z_avg": vz_avg,
        "f0": f0,
        "f_expected": f_exp,
        "lambda0": lam0,
        "lam_theory": lam_th,
        "P_larmor": Pl,
        "P_schwinger": Ps_avg,
        "K_power_eff_rms": K_power_eff,
        "R_avg": traj["avg_radius"],
        "R_max": traj["max_radius"],
        "R_exact": R_ex,
        "circularity": traj["circularity"],
        "pitch": traj["pitch"],
        "P_circ": stokes["P_circ"],
        "P_lin": stokes["P_lin"],
        "peak_amplitude": pulse["peak_amplitude"],
        "avg_fwhm": pulse["avg_fwhm"],
        "repetition_freq": pulse["repetition_freq"],
        "spectral_fwhm_hz": spec["fwhm_hz"],
        "relative_linewidth": spec["relative_linewidth"],
        "spectral_quality_factor": spec["quality_factor"],
        "photon_energy_eV": spec["photon_energy"]["eV"],
        "photon_energy_keV": spec["photon_energy"]["keV"],
        "radiation_H3_over_H1": spec["harmonic_ratios"]["H3_over_H1"],
        "radiation_H5_over_H1": spec["harmonic_ratios"]["H5_over_H1"],
        "H3_over_H1": spec["harmonic_ratios"]["H3_over_H1"],
        "H5_over_H1": spec["harmonic_ratios"]["H5_over_H1"],
        "equivalent_photons": photon_yield["equivalent_photons"],
        "equivalent_photon_rate_s^-1": photon_yield["equivalent_photon_rate_s^-1"],
        "spectral_photons_estimate": spectral_photon_yield["spectral_photons_estimate"],
        "spectral_photon_rate_estimate_s^-1": spectral_photon_yield["spectral_photon_rate_estimate_s^-1"],
        "wiggler_critical_energy_eV": wiggler_critical_energy(und, g_avg)["eV"],
        "frequency_relative_residual": residuals["frequency_relative_residual"],
        "max_transverse_excursion_m": traj_phase.get("max_transverse_excursion_m", np.nan),
        "period_repeatability_rms_m": traj_phase.get("period_repeatability_rms_m", np.nan),
        "orbit_phase_error_rms_rad": traj_phase.get("orbit_phase_error_rms_rad", np.nan),
        "exit_xprime_rad": traj_phase.get("exit_xprime_rad", np.nan),
        "exit_yprime_rad": traj_phase.get("exit_yprime_rad", np.nan),
        "energy_mismatch": energy_check["relative_mismatch"],
        "lost_to_aperture": lost_to_aperture,
        "reached_field_map_end": reached_field_map_end,
        "loss_time_s": loss_time,
        "chi_max": qdiag["chi_max"],
        "g_min": qdiag["g_min"],
        "device_name": und.device_name,
        "Kx": und.K_components(me)["Kx"],
        "Ky": und.K_components(me)["Ky"],
        "MLE": ch["mle"],
        "mle_count": ch["mle_count"],
        "r_diffusion": ch["r_diffusion"],
        "n_poincare": ch["n_poincare"],
        "observer_theta_rad": theta_obs,
        "n_source_samples": int(len(ts)),
    }


def angular_divergence(theta, intensity):
    """Fluence/intensity-weighted 1-D angular mean, RMS divergence, and FWHM."""
    theta = np.asarray(theta, dtype=float)
    raw = np.asarray(intensity, dtype=float)
    if theta.ndim != 1 or raw.ndim != 1 or len(theta) != len(raw):
        raise ValueError("theta and intensity/fluence must be equal-length 1-D arrays")
    valid = np.isfinite(theta) & np.isfinite(raw) & (raw >= 0.0)
    theta = theta[valid]
    I = raw[valid]
    if len(theta) < 3 or np.sum(I) <= 0:
        return {
            "mean_theta_rad": np.nan,
            "rms_divergence_rad": np.nan,
            "fwhm_divergence_rad": np.nan,
        }

    wsum = np.sum(I)
    mean = float(np.sum(theta * I) / wsum)
    rms = float(np.sqrt(np.sum((theta - mean) ** 2 * I) / wsum))

    ip = int(np.argmax(I))
    half = 0.5 * I[ip]
    il = ip
    while il > 0 and I[il] >= half:
        il -= 1
    ir = ip
    while ir < len(I) - 1 and I[ir] >= half:
        ir += 1

    def interp_cross(i0, i1):
        x0, x1 = theta[i0], theta[i1]
        y0, y1 = I[i0], I[i1]
        if abs(y1 - y0) < 1e-300:
            return 0.5 * (x0 + x1)
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    if il == ip or ir == ip:
        fwhm = np.nan
    else:
        left = interp_cross(il, il + 1)
        right = interp_cross(ir - 1, ir)
        fwhm = float(max(right - left, 0.0))

    return {
        "mean_theta_rad": mean,
        "rms_divergence_rad": rms,
        "fwhm_divergence_rad": fwhm,
    }


def radiative_fluence_J_m2(t_obs, E):
    """Far-field electromagnetic energy fluence per detector area.

    Uses the vacuum Poynting flux S = eps0*c*|E|^2 for the transverse
    radiation field and integrates it over observer time.
    """
    t=np.asarray(t_obs,dtype=float)
    field=np.asarray(E,dtype=float)
    if t.ndim!=1 or field.ndim!=2 or field.shape[0]!=len(t) or field.shape[1]!=3:
        raise ValueError("Expected t_obs shape (N,) and E shape (N,3).")
    if len(t)<2:
        return 0.0
    e2=np.sum(field*field,axis=1)
    fn=getattr(np,"trapezoid",None)
    integral=(fn(e2,t) if fn is not None else np.sum(0.5*(e2[1:]+e2[:-1])*np.diff(t)))
    return float(eps_0*c0*integral)


def angular_map_2d(
    res,
    gamma_for_grid=None,
    grid_points=31,
    extent_gamma_theta=4.0,
    observer_distance=100.0,
    n_obs=5000,
):
    """2-D single-electron far-field fluence map.

    Failed pixels are NaN, never silently replaced by zero. A validity mask and
    compact failure list are returned for UI/reporting.
    """
    r_spl,v_spl,a_spl,t0,t1=res["splines"]
    gamma_use=float(gamma_for_grid or res["gamma_avg"])
    u=np.linspace(-extent_gamma_theta,extent_gamma_theta,int(grid_points))
    theta_x=u/gamma_use
    theta_y=u/gamma_use

    shape=(len(theta_y),len(theta_x))
    fluence=np.full(shape,np.nan,dtype=float)
    P_circ=np.full(shape,np.nan,dtype=float)
    P_lin=np.full(shape,np.nan,dtype=float)
    f_peak=np.full(shape,np.nan,dtype=float)
    f_expected_map=np.full(shape,np.nan,dtype=float)
    valid_mask=np.zeros(shape,dtype=bool)
    failures=[]

    for iy,ty in enumerate(theta_y):
        for ix,tx in enumerate(theta_x):
            ro=np.array([
                float(observer_distance)*np.tan(tx),
                float(observer_distance)*np.tan(ty),
                float(observer_distance),
            ])
            f_exp=expected_frequency_from_result(res,ro)
            f_expected_map[iy,ix]=f_exp
            try:
                to_sc,E_sc,tp_sc,_=observer_signal(
                    ro,r_spl,v_spl,a_spl,t0,t1,f_exp,
                    n_obs=int(n_obs),
                    coarse_samples=min(max(700,int(n_obs)//2),1800),
                    min_cycles=10,
                )
                if len(to_sc)<50:
                    raise RuntimeError(f"insufficient observer samples: {len(to_sc)}")
                _,e1,e2=transv_basis(ro)
                sp=get_spec(to_sc,E_sc,e1,e2,f_exp)
                fluence[iy,ix]=radiative_fluence_J_m2(to_sc,E_sc)
                P_circ[iy,ix]=sp["Stokes"]["P_circ"]
                P_lin[iy,ix]=sp["Stokes"]["P_lin"]
                f_peak[iy,ix]=sp["f0"]
                valid_mask[iy,ix]=True
            except Exception as exc:
                failures.append({
                    "iy":int(iy),"ix":int(ix),
                    "theta_x_rad":float(tx),"theta_y_rad":float(ty),
                    "error":f"{type(exc).__name__}: {exc}",
                })

    TX,TY=np.meshgrid(theta_x,theta_y)
    weights=np.where(valid_mask & np.isfinite(fluence) & (fluence>0.0),fluence,0.0)
    wsum=float(np.sum(weights))
    if wsum>0.0:
        mx=float(np.sum(TX*weights)/wsum)
        my=float(np.sum(TY*weights)/wsum)
        sx=float(np.sqrt(np.sum((TX-mx)**2*weights)/wsum))
        sy=float(np.sqrt(np.sum((TY-my)**2*weights)/wsum))
    else:
        mx=my=sx=sy=np.nan

    return {
        "theta_x":theta_x,
        "theta_y":theta_y,
        "gamma_theta":u,
        "fluence_J_m2":fluence,
        # Backward-compatible key; its quantity is now real fluence, not peak |E|^2.
        "intensity":fluence,
        "intensity_quantity":"radiative_fluence_J_m2",
        "P_circ":P_circ,
        "P_lin":P_lin,
        "f_peak_hz":f_peak,
        "f_expected_hz":f_expected_map,
        "valid_mask":valid_mask,
        "failure_count":int(len(failures)),
        "failures":failures,
        "mean_theta_x_rad":mx,
        "mean_theta_y_rad":my,
        "rms_divergence_x_rad":sx,
        "rms_divergence_y_rad":sy,
    }


def angle_scan(res,theta_range,n_obs=8000):
    """1-D far-field scan.

    Columns: theta_rad, f0_Hz, fluence_J_m2, P_circ, P_lin.
    Failed angles are represented by NaN quantities rather than physical zero.
    """
    r_spl,v_spl,a_spl,t0,t1=res["splines"]
    base_obs=np.asarray(res.get("r_obs",[0.0,0.0,100.0]),dtype=float)
    R=abs(float(base_obs[2])) if abs(float(base_obs[2]))>0 else 100.0
    out=[]

    for theta in theta_range:
        ro_sc=np.array([R*np.tan(float(theta)),0.0,R])
        f_exp=expected_frequency_from_result(res,ro_sc)
        try:
            to_sc,E_sc,tp_sc,_=observer_signal(
                ro_sc,r_spl,v_spl,a_spl,t0,t1,f_exp,
                n_obs=int(n_obs),
                coarse_samples=min(max(700,int(n_obs)//2),1800),
                min_cycles=10,
            )
            if len(to_sc)<50:
                raise RuntimeError(f"insufficient observer samples: {len(to_sc)}")
            _,e1,e2=transv_basis(ro_sc)
            sp=get_spec(to_sc,E_sc,e1,e2,f_exp)
            f0=sp["f0"] if sp["f0"]>0 else f_exp
            fluence=radiative_fluence_J_m2(to_sc,E_sc)
            out.append((
                float(theta),float(f0),float(fluence),
                float(sp["Stokes"]["P_circ"]),
                float(sp["Stokes"]["P_lin"]),
            ))
        except Exception:
            out.append((float(theta),np.nan,np.nan,np.nan,np.nan))
    return np.asarray(out,dtype=float)


def chaos_analysis(res, helical_und):
    r = res['r']
    u = res['u']
    g_arr = res['g_arr']
    v_local = u / (g_arr[:, None] * me)
    x, z = r[:, 0], r[:, 2]
    vx = v_local[:, 0]

    lu = helical_und.lambda_u
    z0 = z[0]
    n_min = int((z[0] + 2 * lu - z0) / lu) + 3
    n_max = int((z[-1] - 2 * lu - z0) / lu) - 3
    xs_sec, vxs_sec = [], []
    for n in range(n_min, n_max):
        z_target = z0 + n * lu
        idx_z = np.argmin(np.abs(z - z_target))
        xs_sec.append(x[idx_z])
        vxs_sec.append(vx[idx_z])
    xs_sec = np.array(xs_sec)
    vxs_sec = np.array(vxs_sec)

    N = len(xs_sec)
    mle_sum = 0.0
    mle_count = 0
    eps = 1e-12
    for i in range(max(0, N - 20)):
        dx = xs_sec[i + 1:] - xs_sec[i]
        dvx = vxs_sec[i + 1:] - vxs_sec[i]
        d2 = dx * dx + dvx * dvx
        if len(d2) == 0 or d2.min() < eps:
            continue
        j = np.argmin(d2) + i + 1
        if j >= N - 1:
            continue
        d0 = np.sqrt((xs_sec[j] - xs_sec[i]) ** 2 + (vxs_sec[j] - vxs_sec[i]) ** 2)
        d1 = np.sqrt((xs_sec[j + 1] - xs_sec[i + 1]) ** 2 + (vxs_sec[j + 1] - vxs_sec[i + 1]) ** 2)
        if d0 > eps and d1 > eps:
            mle_sum += np.log(d1 / d0)
            mle_count += 1
    mle = mle_sum / mle_count if mle_count > 0 else 0.0

    rperp = np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)
    r_mean = rperp.mean()
    r_std = rperp.std()
    r_diff = r_std / r_mean if r_mean > 1e-20 else 0.0

    return {
        "poincare_x": xs_sec,
        "poincare_vx": vxs_sec,
        "mle": float(mle),
        "mle_count": int(mle_count),
        "circularity": float(res['traj']['circularity']),
        "r_diffusion": float(r_diff),
        "n_poincare": int(N)
    }

def get_adaptive_gamma_grid(gamma_min=1.25, gamma_max=6.0e4):
    """Adaptive scan grid that resolves both the old low-beta region and beta -> 1.

    gamma_max=1e4 corresponds to beta ~= 0.999999995 and, for the default
    K~=0.70 and lambda_u=5 cm, reaches sub-nm fundamental wavelengths.
    """
    # Dense linear sampling where the old study lived (roughly 0.60c--0.95c).
    g_low = np.linspace(gamma_min, 3.25, 28)
    # Log sampling is much more efficient once beta visually saturates near 1.
    g_mid = np.geomspace(3.25, 100.0, 24)
    # Keep the original dense ultra-relativistic range through gamma=1e4,
    # then append only a few gamma-edge points to minimize extra runtime.
    g_high = np.geomspace(100.0, 1.0e4, 32)
    g_gamma_edge = np.array([1.5e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4, 6.0e4])
    g_all = np.concatenate([g_low, g_mid, g_high, g_gamma_edge])
    return np.unique(g_all[(g_all >= gamma_min) & (g_all <= gamma_max)])



# =============================================================================
# RADIA / 3-D FIELD-MAP INTEGRATION
# =============================================================================
# FIELD_MODEL is now a third top-level choice alongside DEVICE_PRESET and
# SCAN_PRESET.
#
#   "analytic"        -> original V11 realistic analytic field
#   "radia_generated" -> build a permanent-magnet model with the official
#                        locally compiled RADIA engine, sample a 3-D field map,
#                        then use interpolation during particle tracking
#   "radia_csv"       -> load a 3-D map exported by the RADIA GUI
#
# The default of THIS integrated file is RADIA-generated, because this file is
# specifically the V11+RADIA edition. Set it back to "analytic" at any time.
FIELD_MODEL = "radia_generated"

RADIA_PYTHONPATH = os.environ.get(
    "RADIA_PYTHONPATH",
    os.path.expanduser("~/Desktop/Radia-master/cpp/gcc")
)

# Used only by FIELD_MODEL="radia_csv".
RADIA_FIELD_CSV = os.path.expanduser("~/Desktop/radia_helical_3d_field.csv")
RADIA_CSV_SHIFT_Z_TO_ZERO = True

# Fixed RADIA generation settings. These are deliberately conservative so the
# map is usable on a laptop. Increase Nx/Ny/Nz only after validating convergence.
RADIA_TARGET_B0_T = 0.15
RADIA_MAGNETIZATION_GUESS_T = 1.20
RADIA_GAP_M = 0.012
RADIA_BLOCK_WIDTH_M = 0.040
RADIA_BLOCK_HEIGHT_M = 0.015
RADIA_MAP_X_HALF_M = 0.003
RADIA_MAP_Y_HALF_M = 0.003
RADIA_MAP_NX = 7
RADIA_MAP_NY = 7
RADIA_MAP_SAMPLES_PER_PERIOD = 24
RADIA_APPLY_ENGINEERING_ERRORS = True
RADIA_ERROR_SEED = 20260820

# Individual RADIA prototype error sources.
# These switches can be changed directly in Python or overridden temporarily by
# the GUI. All magnitudes are intentionally modest engineering-scale prototype
# values rather than universal machine tolerances.
RADIA_ERROR_CONFIG = {
    "field_amplitude": {
        "enabled": True,
        "rms_fraction": 0.002,          # 0.2% block magnetization amplitude RMS
    },
    "longitudinal_position": {
        "enabled": True,
        "rms_m": 20e-6,                 # 20 um block z-position RMS
    },
    "transverse_position": {
        "enabled": True,
        "rms_m": 10e-6,                 # 10 um x/y block placement RMS
    },
    "magnetization_angle": {
        "enabled": True,
        "rms_rad": 0.5e-3,              # 0.5 mrad magnetization-direction RMS
    },
    "gap_asymmetry": {
        "enabled": True,
        "rms_m": 10e-6,                 # 10 um local bank normal-position RMS
    },
    "bank_strength_imbalance": {
        "enabled": True,
        "rms_fraction": 0.001,           # 0.1% pair/bank strength imbalance RMS
    },
}

def default_radia_error_switches():
    return {k: bool(v["enabled"]) for k, v in RADIA_ERROR_CONFIG.items()}

_RADIA_DEVICE_CACHE = {}


def load_radia_module():
    """Import the user's locally compiled official RADIA Python extension."""
    first_error = None
    second_error = None
    try:
        import radia as rad
        return rad
    except Exception as exc:
        first_error = exc

    if RADIA_PYTHONPATH and os.path.isdir(RADIA_PYTHONPATH):
        if RADIA_PYTHONPATH not in sys.path:
            sys.path.insert(0, RADIA_PYTHONPATH)
        try:
            import radia as rad
            return rad
        except Exception as exc:
            second_error = exc

    raise RuntimeError(
        "RADIA cannot be imported. Expected the compiled module in "
        f"{RADIA_PYTHONPATH!r}. Normal import error: {first_error!r}. "
        f"Explicit-path import error: {second_error!r}."
    )


class FieldMapInsertionDevice:
    """Insertion-device interface backed by a regular 3-D B(x,y,z) field map.

    Coordinates are SI metres and field values are tesla. The class implements
    the same methods/attributes used by the V11 Lorentz/radiation solver, so
    the rest of the physics code does not need a separate tracking engine.
    """

    uses_real_end_fields = True

    def __init__(
        self,
        x_m, y_m, z_m, bx_T, by_T, bz_T,
        lambda_u,
        device_name="field_map",
        handedness=1,
        aperture_radius=None,
        source_label="3-D field map",
        metadata=None,
    ):
        self.x_grid = np.asarray(x_m, dtype=float)
        self.y_grid = np.asarray(y_m, dtype=float)
        self.z_grid = np.asarray(z_m, dtype=float)
        shape = (len(self.x_grid), len(self.y_grid), len(self.z_grid))
        self._bx_arr = np.asarray(bx_T, dtype=float).reshape(shape)
        self._by_arr = np.asarray(by_T, dtype=float).reshape(shape)
        self._bz_arr = np.asarray(bz_T, dtype=float).reshape(shape)

        interp_kw = dict(bounds_error=False, fill_value=0.0)
        self._bx = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid), self._bx_arr, **interp_kw
        )
        self._by = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid), self._by_arr, **interp_kw
        )
        self._bz = RegularGridInterpolator(
            (self.x_grid, self.y_grid, self.z_grid), self._bz_arr, **interp_kw
        )

        self.lambda_u = float(lambda_u)
        self.k_u = 2.0 * np.pi / self.lambda_u
        self.device_name = str(device_name)
        self.handedness = int(handedness)
        self.source_label = str(source_label)
        self.metadata = {} if metadata is None else dict(metadata)

        self.device_length = float(max(self.z_grid) - max(0.0, min(self.z_grid)))
        if "device_length_m" in self.metadata:
            self.device_length = float(self.metadata["device_length_m"])
        self.device_n_periods = max(
            1, int(round(self.device_length / self.lambda_u))
        )

        # The map itself is the field model; no analytic fringe envelope/errors
        # are layered on top.
        self.use_fringe_fields = False
        self.fringe_periods = 0.0
        self.field_rms = 0.0
        self.position_rms = 0.0
        self.phase_mismatch = 0.0
        self.transverse_imbalance = 1.0
        self.bx_scale = 1.0
        self.by_scale = 1.0

        if aperture_radius is None:
            aperture_radius = 0.90 * min(
                max(abs(self.x_grid[0]), abs(self.x_grid[-1])),
                max(abs(self.y_grid[0]), abs(self.y_grid[-1])),
            )
        self.aperture_radius = float(aperture_radius)

        # Characterize the periodic central field separately from fringe/end fields.
        ix = int(np.argmin(np.abs(self.x_grid)))
        iy = int(np.argmin(np.abs(self.y_grid)))
        bx_axis = self._bx_arr[ix, iy, :]
        by_axis = self._by_arr[ix, iy, :]
        bz_axis = self._bz_arr[ix, iy, :]

        self.Bx_peak_global = float(np.max(np.abs(bx_axis)))
        self.By_peak_global = float(np.max(np.abs(by_axis)))
        self.Bz_peak_global = float(np.max(np.abs(bz_axis)))
        self.Btrans_peak_global = float(np.max(np.hypot(bx_axis, by_axis)))

        geom_edges = self.metadata.get("geometry_z_edges_m")
        if isinstance(geom_edges, (list, tuple)) and len(geom_edges) == 2:
            geom_lo, geom_hi = map(float, geom_edges)
            center = 0.5 * (geom_lo + geom_hi)
            geom_span = max(geom_hi - geom_lo, self.lambda_u)
        else:
            center = 0.5 * (float(self.z_grid[0]) + float(self.z_grid[-1]))
            geom_span = max(float(self.z_grid[-1] - self.z_grid[0]), self.lambda_u)

        char_periods = min(3.0, max(1.0, geom_span / self.lambda_u - 2.0))
        half = 0.5 * char_periods * self.lambda_u
        zlo = max(float(self.z_grid[0]), center - half)
        zhi = min(float(self.z_grid[-1]), center + half)
        mask = (self.z_grid >= zlo) & (self.z_grid <= zhi)
        if np.count_nonzero(mask) < 8:
            mask = np.ones_like(self.z_grid, dtype=bool)
            zlo, zhi = float(self.z_grid[0]), float(self.z_grid[-1])

        zc = self.z_grid[mask]
        bxc = bx_axis[mask]
        byc = by_axis[mask]

        def fundamental_amplitude(signal):
            M = np.column_stack([
                np.sin(self.k_u * zc),
                np.cos(self.k_u * zc),
                np.ones_like(zc),
            ])
            coef, *_ = np.linalg.lstsq(M, np.asarray(signal, dtype=float), rcond=None)
            return float(np.hypot(coef[0], coef[1]))

        self.Bx_fundamental = fundamental_amplitude(bxc)
        self.By_fundamental = fundamental_amplitude(byc)
        self.Btrans_peak_central = float(np.max(np.hypot(bxc, byc)))
        self.Bx_peak = self.Bx_fundamental
        self.By_peak = self.By_fundamental
        self.B0 = self.Btrans_peak_central
        if self.B0 <= 0:
            self.B0 = self.Btrans_peak_global

        self.metadata["characterization_z_range_m"] = [float(zlo), float(zhi)]
        self.metadata["Bx_fundamental_T"] = self.Bx_fundamental
        self.metadata["By_fundamental_T"] = self.By_fundamental
        self.metadata["Btrans_peak_central_T"] = self.Btrans_peak_central
        self.metadata["Btrans_peak_global_T"] = self.Btrans_peak_global

    def B(self, r):
        arr = np.asarray(r, dtype=float)
        scalar = arr.ndim == 1
        pts = arr.reshape(1, 3) if scalar else arr
        b = np.column_stack([
            self._bx(pts),
            self._by(pts),
            self._bz(pts),
        ])
        return b[0] if scalar else b

    def K(self, m):
        return qe * self.B0 * self.lambda_u / (2.0 * np.pi * m * c0)

    def K_components(self, m):
        fac = qe * self.lambda_u / (2.0 * np.pi * m * c0)
        # Bx drives y motion and By drives x motion; the generalized resonance
        # factor only needs their squared amplitudes.
        K_Bx = fac * self.Bx_peak
        K_By = fac * self.By_peak
        return {
            "K0": self.K(m),
            "Kx": float(K_Bx),
            "Ky": float(K_By),
            "K_eff_rms": float(np.sqrt(0.5 * (K_Bx*K_Bx + K_By*K_By))),
        }

    def inside_aperture(self, r):
        a = np.asarray(r, dtype=float)
        if a.ndim == 1:
            return bool(np.hypot(a[0], a[1]) <= self.aperture_radius)
        return np.hypot(a[:, 0], a[:, 1]) <= self.aperture_radius

    def error_summary(self):
        return {
            "field_model": "3-D map",
            "source": self.source_label,
            "device": self.device_name,
            "B0_central_peak_T": self.B0,
            "Bx_fundamental_T": self.Bx_fundamental,
            "By_fundamental_T": self.By_fundamental,
            "Btrans_peak_global_T": self.Btrans_peak_global,
            "lambda_u_m": self.lambda_u,
            "grid_shape": (
                len(self.x_grid), len(self.y_grid), len(self.z_grid)
            ),
            "map_x_m": (float(self.x_grid[0]), float(self.x_grid[-1])),
            "map_y_m": (float(self.y_grid[0]), float(self.y_grid[-1])),
            "map_z_m": (float(self.z_grid[0]), float(self.z_grid[-1])),
            "metadata": self.metadata,
        }


def _field_map_from_dataframe(
    df,
    lambda_u,
    device_name,
    handedness=1,
    shift_z_to_zero=False,
    source_label="CSV field map",
):
    required = {"x_m", "y_m", "z_m", "Bx_T", "By_T", "Bz_T"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Field-map CSV missing columns: {sorted(missing)}")

    d = df.copy()
    if shift_z_to_zero:
        d["z_m"] = d["z_m"] - float(d["z_m"].min())

    x = np.sort(d["x_m"].unique())
    y = np.sort(d["y_m"].unique())
    z = np.sort(d["z_m"].unique())
    idx = pd.MultiIndex.from_product([x, y, z], names=["x_m", "y_m", "z_m"])
    ordered = d.set_index(["x_m", "y_m", "z_m"]).reindex(idx)
    if ordered[["Bx_T", "By_T", "Bz_T"]].isna().any().any():
        raise ValueError("Field map is not a complete regular x-y-z grid.")

    shape = (len(x), len(y), len(z))
    return FieldMapInsertionDevice(
        x, y, z,
        ordered["Bx_T"].to_numpy().reshape(shape),
        ordered["By_T"].to_numpy().reshape(shape),
        ordered["Bz_T"].to_numpy().reshape(shape),
        lambda_u=float(lambda_u),
        device_name=device_name,
        handedness=handedness,
        source_label=source_label,
        metadata={"device_length_m": float(z.max() - z.min())},
    )


def load_radia_csv_device(
    csv_path=RADIA_FIELD_CSV,
    lambda_u=0.05,
    device_name="helical",
    handedness=1,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"RADIA_FIELD_CSV not found: {csv_path}. Export a 3-D field map "
            "from the RADIA GUI first or change RADIA_FIELD_CSV."
        )
    df = pd.read_csv(csv_path)
    return _field_map_from_dataframe(
        df,
        lambda_u=lambda_u,
        device_name=device_name,
        handedness=handedness,
        shift_z_to_zero=RADIA_CSV_SHIFT_Z_TO_ZERO,
        source_label=os.path.basename(csv_path),
    )


def _radia_add_block(rad, handles, center_mm, size_mm, magnetization_T):
    handles.append(
        rad.ObjRecMag(
            [float(v) for v in center_mm],
            [float(v) for v in size_mm],
            [float(v) for v in magnetization_T],
        )
    )



def _rotate_vector_small(v, dtheta_x=0.0, dtheta_y=0.0, dtheta_z=0.0):
    """First-order small-angle rotation of a 3-vector."""
    v = np.asarray(v, dtype=float)
    omega = np.array([dtheta_x, dtheta_y, dtheta_z], dtype=float)
    return v + np.cross(omega, v)


def _build_radia_geometry(
    device_name,
    lambda_u,
    n_periods,
    magnetization_T,
    gap_m,
    width_m,
    height_m,
    apply_errors=True,
    error_switches=None,
):
    """Build the same transparent permanent-magnet prototype used by the GUI."""
    if device_name not in {"helical", "left_helical", "planar"}:
        raise ValueError(
            "RADIA-generated mode currently supports helical, left_helical, "
            "and planar. Use analytic mode for elliptical/variable/wiggler."
        )

    rad = load_radia_module()
    if hasattr(rad, "UtiDelAll"):
        rad.UtiDelAll()

    rng = np.random.default_rng(RADIA_ERROR_SEED)
    if error_switches is None:
        error_switches = default_radia_error_switches()
    else:
        base = default_radia_error_switches()
        base.update({k: bool(v) for k, v in error_switches.items() if k in base})
        error_switches = base
    if not apply_errors:
        error_switches = {k: False for k in error_switches}

    period_mm = lambda_u * 1e3
    gap_mm = gap_m * 1e3
    width_mm = width_m * 1e3
    height_mm = height_m * 1e3
    block_len = period_mm / 4.0
    n_blocks = int(n_periods) * 4
    d = 0.5 * gap_mm + 0.5 * height_mm
    handles = []

    def add_pair(axis, z_shift_mm=0.0, handed=1):
        # One fixed pair-level imbalance value helps isolate systematic bank
        # imbalance from block-to-block random amplitude errors.
        pair_imbalance = 0.0
        if error_switches.get("bank_strength_imbalance", False):
            pair_imbalance = rng.normal(
                0.0, RADIA_ERROR_CONFIG["bank_strength_imbalance"]["rms_fraction"]
            )

        for j in range(n_blocks):
            z = 0.5*block_len + j*block_len + z_shift_mm
            phi = 2.0*np.pi*j/4.0

            amp_scale = 1.0
            if error_switches.get("field_amplitude", False):
                amp_scale += rng.normal(
                    0.0, RADIA_ERROR_CONFIG["field_amplitude"]["rms_fraction"]
                )

            z_err_mm = 0.0
            if error_switches.get("longitudinal_position", False):
                z_err_mm = 1e3 * rng.normal(
                    0.0, RADIA_ERROR_CONFIG["longitudinal_position"]["rms_m"]
                )

            tx_err_mm = 0.0
            ty_err_mm = 0.0
            if error_switches.get("transverse_position", False):
                sxy = 1e3 * RADIA_ERROR_CONFIG["transverse_position"]["rms_m"]
                tx_err_mm = rng.normal(0.0, sxy)
                ty_err_mm = rng.normal(0.0, sxy)

            gap_err_mm = 0.0
            if error_switches.get("gap_asymmetry", False):
                gap_err_mm = 1e3 * rng.normal(
                    0.0, RADIA_ERROR_CONFIG["gap_asymmetry"]["rms_m"]
                )

            mn = magnetization_T * amp_scale * np.cos(phi)
            mz = magnetization_T * amp_scale * np.sin(phi)
            zc = z + z_err_mm

            if axis == "y":
                size = (width_mm, height_mm, block_len)
                Mp = np.array((0.0, +mn*(1.0+pair_imbalance), mz), dtype=float)
                Mm = np.array((0.0, -mn*(1.0-pair_imbalance), mz), dtype=float)

                if error_switches.get("magnetization_angle", False):
                    sr = RADIA_ERROR_CONFIG["magnetization_angle"]["rms_rad"]
                    Mp = _rotate_vector_small(
                        Mp, *rng.normal(0.0, sr, size=3)
                    )
                    Mm = _rotate_vector_small(
                        Mm, *rng.normal(0.0, sr, size=3)
                    )

                _radia_add_block(
                    rad, handles,
                    (tx_err_mm, +d+ty_err_mm+gap_err_mm, zc),
                    size, Mp
                )
                _radia_add_block(
                    rad, handles,
                    (tx_err_mm, -d+ty_err_mm-gap_err_mm, zc),
                    size, Mm
                )
            else:
                size = (height_mm, width_mm, block_len)
                s = handed
                Mp = np.array((+s*mn*(1.0+pair_imbalance), 0.0, mz), dtype=float)
                Mm = np.array((-s*mn*(1.0-pair_imbalance), 0.0, mz), dtype=float)

                if error_switches.get("magnetization_angle", False):
                    sr = RADIA_ERROR_CONFIG["magnetization_angle"]["rms_rad"]
                    Mp = _rotate_vector_small(
                        Mp, *rng.normal(0.0, sr, size=3)
                    )
                    Mm = _rotate_vector_small(
                        Mm, *rng.normal(0.0, sr, size=3)
                    )

                _radia_add_block(
                    rad, handles,
                    (+d+tx_err_mm+gap_err_mm, ty_err_mm, zc),
                    size, Mp
                )
                _radia_add_block(
                    rad, handles,
                    (-d+tx_err_mm-gap_err_mm, ty_err_mm, zc),
                    size, Mm
                )

    add_pair("y")
    if device_name in {"helical", "left_helical"}:
        handed = 1 if device_name == "helical" else -1
        add_pair("x", z_shift_mm=0.25*period_mm, handed=handed)

    return rad, rad.ObjCnt(handles)


def _sample_radia_axis_peak(rad, obj, lambda_u, n_periods, samples_per_period=24):
    L = n_periods * lambda_u
    z = np.linspace(0.0, L, max(101, n_periods*samples_per_period+1))
    B = np.array([
        rad.Fld(obj, "b", [0.0, 0.0, float(zi*1e3)])
        for zi in z
    ], dtype=float)
    return float(np.max(np.hypot(B[:,0], B[:,1])))



def generate_radia_field_device(
    device_name="helical",
    lambda_u=0.05,
    n_periods=100,
    handedness=1,
    target_B0_T=RADIA_TARGET_B0_T,
    error_switches=None,
    radia_options=None,
):
    """Generate a strict RADIA 3-D field map and adapt it to V11."""
    if error_switches is None:
        error_switches=default_radia_error_switches()
    ro = {} if radia_options is None else dict(radia_options)

    error_key=tuple(sorted((str(k),bool(v)) for k,v in error_switches.items()))
    cache_key=(
        "v5_bridge",device_name,float(lambda_u),int(n_periods),float(target_B0_T),
        RADIA_MAP_NX,RADIA_MAP_NY,RADIA_MAP_SAMPLES_PER_PERIOD,
        RADIA_APPLY_ENGINEERING_ERRORS,error_key,
        tuple(sorted((str(k), repr(v)) for k,v in ro.items())),
    )
    if cache_key in _RADIA_DEVICE_CACHE:
        return _RADIA_DEVICE_CACHE[cache_key]

    from v11_radia_backend_v8 import build_radia_map

    switches=(
        dict(error_switches)
        if RADIA_APPLY_ENGINEERING_ERRORS
        else {k:False for k in default_radia_error_switches()}
    )
    fmap=build_radia_map(
        device_name=device_name,
        lambda_u_m=float(ro.get("lambda_u_m", lambda_u)),
        n_periods=int(n_periods),
        target_B0_T=float(target_B0_T),
        gap_m=float(ro.get("gap_m", RADIA_GAP_M)),
        block_width_m=float(ro.get(
            "block_width_m",
            0.010 if device_name in ("helical","left_helical","elliptical") else RADIA_BLOCK_WIDTH_M
        )),
        block_height_m=float(ro.get("block_height_m", RADIA_BLOCK_HEIGHT_M)),
        x_half_m=float(ro.get("x_half_m", RADIA_MAP_X_HALF_M)),
        y_half_m=float(ro.get("y_half_m", RADIA_MAP_Y_HALF_M)),
        nx=int(ro.get("nx", RADIA_MAP_NX)),
        ny=int(ro.get("ny", RADIA_MAP_NY)),
        samples_per_period=int(ro.get("samples_per_period", RADIA_MAP_SAMPLES_PER_PERIOD)),
        field_margin_periods=float(ro.get("field_margin_periods", 1.0)),
        error_switches=switches,
        error_config=ro.get("error_config", RADIA_ERROR_CONFIG),
        error_seed=int(ro.get("error_seed", RADIA_ERROR_SEED)),
        material_mode=str(ro.get("material_mode", "Fixed remanence")),
        mu_parallel=float(ro.get("mu_parallel", 1.05)),
        mu_perpendicular=float(ro.get("mu_perpendicular", 1.05)),
        segmentation=tuple(ro.get("segmentation", (1,1,1))),
        ellipticity=float(ro.get("ellipticity", 0.50)),
        apple_phase_deg=float(ro.get("apple_phase_deg", 90.0)),
        apple_shift_mode=str(ro.get("apple_shift_mode", "Antiparallel")),
    )

    dev=FieldMapInsertionDevice(
        fmap["x_m"],fmap["y_m"],fmap["z_m"],
        fmap["Bx_T"],fmap["By_T"],fmap["Bz_T"],
        lambda_u=float(ro.get("lambda_u_m", lambda_u)),
        device_name=device_name,
        handedness=handedness,
        aperture_radius=0.90*min(
            float(ro.get("x_half_m", RADIA_MAP_X_HALF_M)),
            float(ro.get("y_half_m", RADIA_MAP_Y_HALF_M)),
        ),
        source_label="RADIA Magnet Studio strict 3-D map",
        metadata=fmap["metadata"],
    )
    dev.metadata["magnet_block_count"]=len(fmap["blocks"])
    dev.metadata["tracking_time_safety_factor"]=float(ro.get("tracking_time_safety_factor", 1.10))
    _RADIA_DEVICE_CACHE[cache_key]=dev
    return dev


def export_device_field_map_csv(device, csv_path):
    """Export any FieldMapInsertionDevice in the GUI-compatible SI format."""
    if not isinstance(device, FieldMapInsertionDevice):
        raise TypeError("CSV export requires a FieldMapInsertionDevice.")
    X, Y, Z = np.meshgrid(
        device.x_grid, device.y_grid, device.z_grid, indexing="ij"
    )
    df = pd.DataFrame({
        "x_m": X.ravel(),
        "y_m": Y.ravel(),
        "z_m": Z.ravel(),
        "Bx_T": device._bx_arr.ravel(),
        "By_T": device._by_arr.ravel(),
        "Bz_T": device._bz_arr.ravel(),
    })
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------- Insertion-device presets ----------------
# IMPORTANT: The default stays exactly on the original research path:
# right-handed helical undulator + gamma/velocity scan.
DEVICE_PRESET = "helical"

DEVICE_PRESETS = {
    "helical": {
        "device_name": "helical",
        "bx_scale": 1.0,
        "by_scale": 1.0,
        "handedness": 1,
        "transverse_imbalance": 1.0,
        "phase_mismatch": 0.0,
        "wiggler_K": None,
    },
    "left_helical": {
        "device_name": "left_helical",
        "bx_scale": 1.0,
        "by_scale": 1.0,
        "handedness": -1,
        "transverse_imbalance": 1.0,
        "phase_mismatch": 0.0,
        "wiggler_K": None,
    },
    "planar": {
        "device_name": "planar",
        "bx_scale": 0.0,
        "by_scale": 1.0,
        "handedness": 1,
        "transverse_imbalance": 1.0,
        "phase_mismatch": 0.0,
        "wiggler_K": None,
    },
    "elliptical": {
        "device_name": "elliptical",
        "bx_scale": 1.0,
        "by_scale": 0.5,
        "handedness": 1,
        "transverse_imbalance": 1.0,
        "phase_mismatch": 0.0,
        "wiggler_K": None,
    },
    "variable_polarization": {
        "device_name": "variable_polarization",
        "bx_scale": 1.0,
        "by_scale": 0.75,
        "handedness": 1,
        "transverse_imbalance": 1.0,
        "phase_mismatch": 0.0,
        "wiggler_K": None,
    },
    "apple2": {
        "device_name": "apple2",
        "bx_scale": 1.0,
        "by_scale": 1.0,
        "handedness": 1,
        "transverse_imbalance": 1.0,
        "phase_mismatch": 0.0,
        "wiggler_K": None,
    },
    "wiggler": {
        "device_name": "wiggler",
        "bx_scale": 0.0,
        "by_scale": 1.0,
        "handedness": 1,
        "transverse_imbalance": 1.0,
        "phase_mismatch": 0.0,
        "wiggler_K": 5.0,
    },
}

# Optional changes applied on top of the chosen preset.
# Example:
# PRESET_OVERRIDES = {"by_scale": 0.30, "phase_mismatch": np.deg2rad(15)}
PRESET_OVERRIDES = {}


def get_device_preset(name=DEVICE_PRESET, overrides=None):
    if name not in DEVICE_PRESETS:
        raise ValueError(
            f"Unknown DEVICE_PRESET={name!r}; available presets: "
            + ", ".join(DEVICE_PRESETS)
        )
    out = dict(DEVICE_PRESETS[name])
    if overrides:
        out.update(overrides)
    return out


def list_device_presets():
    return tuple(DEVICE_PRESETS.keys())



# ---------------- Scan / analysis / error presets ----------------
# Default remains the original research question: helical device, scan gamma.
SCAN_PRESET = "gamma"
ANALYSIS_PRESET = "full"
# One realistic default error configuration.
# These are illustrative engineering-level imperfections, not a universal
# specification for every real undulator.
REALISTIC_ERRORS = {
    "field_rms": 0.002,
    "position_rms": 20e-6,
    "transverse_imbalance": 0.995,
    "phase_mismatch": np.deg2rad(0.5),
}



# ---------------- Realistic single-electron defaults ----------------
APPLY_REALISTIC_ERRORS = True
REALISTIC_FIELD_CONFIG = {
    # Only the original low odd harmonics are retained: fundamental + 3rd + 5th.
    "h1": 0.03,   # magnetic_h3 amplitude
    "n1": 3,
    "h2": 0.005,  # magnetic_h5 amplitude
    "n2": 5,

    # Reproducible manufacturing/alignment perturbations.
    "field_rms": 0.002,                # 0.2% RMS period-to-period amplitude error
    "position_rms": 20e-6,             # 20 micrometres RMS longitudinal period error
    "transverse_imbalance": 0.995,     # 0.5% By/Bx imbalance
    "phase_mismatch": np.deg2rad(0.5), # Bx/By phase mismatch
    "n_error_periods": 128,
    "error_seed": 20260820,

    # Finite device / end-field realism.
    "device_n_periods": 100,
    "use_fringe_fields": False,  # deliberately OFF by default; use field maps for realistic ends
    "fringe_periods": 1.0,

    # Circular transverse aperture. Set None to disable beam-loss termination.
    "aperture_radius": 0.03,
}

def make_default_undulator(
    realistic=APPLY_REALISTIC_ERRORS,
    preset=DEVICE_PRESET,
    preset_overrides=None,
    field_model=None,
    n_periods=None,
    radia_csv_path=None,
    radia_csv_lambda_u=0.05,
    error_switches=None,
    radia_options=None,
):
    """Construct a V11 insertion device from analytic or RADIA fields."""
    fm = FIELD_MODEL if field_model is None else str(field_model)
    nper = (
        int(REALISTIC_FIELD_CONFIG["device_n_periods"])
        if n_periods is None else int(n_periods)
    )
    overrides = PRESET_OVERRIDES if preset_overrides is None else preset_overrides
    pset = get_device_preset(preset, overrides=overrides)

    if fm == "radia_generated":
        ro = {} if radia_options is None else dict(radia_options)
        lambda_u_generated = float(ro.get("lambda_u_m", 0.05))
        target = ro.get("target_B0_T", None)
        if target is None:
            if pset.get("wiggler_K") is not None:
                target = B0_from_K(pset["wiggler_K"], lambda_u_generated)
            else:
                target = RADIA_TARGET_B0_T
        return generate_radia_field_device(
            device_name=pset["device_name"],
            lambda_u=lambda_u_generated,
            n_periods=nper,
            handedness=pset["handedness"],
            target_B0_T=float(target),
            error_switches=error_switches,
            radia_options=ro,
        )

    if fm == "radia_csv":
        path = RADIA_FIELD_CSV if radia_csv_path is None else radia_csv_path
        return load_radia_csv_device(
            path,
            lambda_u=float(radia_csv_lambda_u),
            device_name=pset["device_name"],
            handedness=pset["handedness"],
        )

    if fm != "analytic":
        raise ValueError(
            f"Unknown FIELD_MODEL={fm!r}; choose analytic, radia_generated, or radia_csv."
        )

    cfg = REALISTIC_FIELD_CONFIG.copy()
    cfg["device_n_periods"] = nper
    cfg["device_length"] = nper * cfg.get("lambda_u", 0.05) if "device_length" in cfg else None
    cfg.pop("device_length", None)

    cfg.update({
        "bx_scale": pset["bx_scale"],
        "by_scale": pset["by_scale"],
        "device_name": pset["device_name"],
    })

    if realistic:
        cfg.update(REALISTIC_ERRORS)
    else:
        cfg.update({
            "field_rms": 0.0,
            "position_rms": 0.0,
            "transverse_imbalance": 1.0,
            "phase_mismatch": 0.0,
            "use_fringe_fields": False,
            "aperture_radius": None,
        })

    und = UndHel(
        0.15,
        0.05,
        handedness=pset["handedness"],
        **cfg,
    )

    if pset.get("wiggler_K") is not None:
        und.B0 = B0_from_K(pset["wiggler_K"], und.lambda_u)

    return und


# ---------------- Lightweight numerical validation suite ----------------
def run_orbit_only(
    und, gamma0, n_periods=20, pts_per_period=96,
    injection=None, rtol=1e-9, atol=1e-11
):
    """Cheap trajectory-only solve for convergence checks."""
    K = und.K(me)
    ku = und.k_u
    state0, init_meta = make_initial_state_device(gamma0, und, injection=injection)
    vz0 = init_meta["vz_initial"]
    t_span = simulation_span_for_device(
        gamma0, und, n_periods=n_periods
    )
    npts = samples_for_periods(
        n_periods, pts_per_period=pts_per_period,
        min_pts=max(500, n_periods * pts_per_period),
        max_pts=max(2000, n_periods * pts_per_period + 1)
    )
    t_eval = np.linspace(t_span[0], t_span[1], npts)
    sol = solve_ivp(
        rhs_lorentz, t_span, state0,
        args=(und, me, -qe),
        t_eval=t_eval, method="RK45",
        rtol=rtol, atol=atol,
        events=(aperture_event, field_map_end_event)
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    ts, sol_y = solution_arrays_with_terminal_sample(sol)
    r = sol_y[:3].T
    u = sol_y[3:].T
    lost_to_aperture = bool(sol.t_events and len(sol.t_events[0]) > 0)
    reached_field_map_end = bool(
        getattr(und, "uses_real_end_fields", False)
        and len(sol.t_events) > 1 and len(sol.t_events[1]) > 0
    )
    if getattr(und, "uses_real_end_fields", False) and not lost_to_aperture and not reached_field_map_end:
        raise RuntimeError("Orbit-only real-map integration did not reach exact z_end.")
    gam = np.sqrt(1.0 + np.sum(u*u, axis=1)/(me*me*c0*c0))
    _, vz_avg = steady_vals(ts, u, me)
    traj = orbit_data(ts, r, und, vz_avg)
    P = instant_P(r, u, und, me, -qe)
    echeck = energy_accounting(ts, gam, P, me)
    return {
        "pts_per_period": int(pts_per_period),
        "rtol": float(rtol),
        "atol": float(atol),
        "gamma_final": float(gam[-1]),
        "avg_radius_m": traj["avg_radius"],
        "pitch_m": traj["pitch"],
        "circularity": traj["circularity"],
        "energy_mismatch": echeck["relative_mismatch"],
        "lost_to_aperture": lost_to_aperture,
        "reached_field_map_end": reached_field_map_end,
    }


def convergence_suite(
    gamma0=100.0, n_periods=20,
    pts_list=(48, 96, 192),
    tolerances=((1e-8, 1e-10), (1e-9, 1e-11), (1e-10, 1e-12)),
    realistic=True
):
    """Return convergence data without running the expensive LW/FFT detector stage."""
    und = make_default_undulator(realistic=realistic)
    sampling = [
        run_orbit_only(
            und, gamma0, n_periods=n_periods,
            pts_per_period=p, rtol=1e-9, atol=1e-11
        )
        for p in pts_list
    ]
    tolerance = [
        run_orbit_only(
            und, gamma0, n_periods=n_periods,
            pts_per_period=96, rtol=rt, atol=at
        )
        for rt, at in tolerances
    ]
    return {"sampling": sampling, "tolerance": tolerance}



# ---------------- Additional analysis scans ----------------

def wiggler_critical_energy(device, gamma, m_part=me, charge_mag=qe):
    """Approximate peak-field synchrotron critical energy for wiggler-like motion."""
    Bpeak = abs(device.B0) * max(
        abs(device.bx_scale),
        abs(device.by_scale * device.transverse_imbalance),
    )
    omega_c = 1.5 * (float(gamma) ** 2) * charge_mag * Bpeak / m_part
    E_J = hbar_planck * omega_c
    return {
        "J": float(E_J),
        "eV": float(E_J / eV_J),
        "keV": float(E_J / eV_J / 1e3),
    }


def B0_from_K(K_target, lambda_u, m_part=me, charge_mag=qe):
    """Convert target undulator K to B0 for the current analytic convention."""
    return float(K_target) * 2.0 * np.pi * m_part * c0 / (charge_mag * lambda_u)


def k_scan(
    gamma0,
    K_values=(0.2, 0.4, 0.7, 1.0, 1.5),
    n_periods=50,
    r_obs=np.array([0.0, 0.0, 100.0]),
    realistic=True,
    n_base=None,
):
    """Full single-electron radiation scan versus K.

    K is changed by adjusting B0 while keeping lambda_u fixed.
    """
    results = []
    for K_target in K_values:
        und = make_default_undulator(realistic=realistic)
        und.device_n_periods = int(n_periods)
        und.device_length = n_periods * und.lambda_u
        und.B0 = B0_from_K(K_target, und.lambda_u)
        if ideal_beta_z_device(gamma0, und) <= 0.0:
            results.append({
                "K": float(K_target),
                "valid": False,
                "reason": "gamma too low for requested K",
            })
            continue

        t_span = simulation_span_for_device(
            gamma0, und, n_periods=n_periods
        )
        if n_base is None:
            n_use = samples_for_periods(n_periods, pts_per_period=96)
        else:
            n_use = int(n_base)
        res = run_sim(
            und,
            beta_from_gamma(gamma0) * c0,
            t_span,
            np.asarray(r_obs, dtype=float),
            n_base=n_use,
            gamma0_input=gamma0,
        )
        results.append({
            "K": float(K_target),
            "B0_T": float(und.B0),
            "valid": True,
            "f0_Hz": float(res["f0"]),
            "lambda_m": float(c0 / res["f0"]) if res["f0"] > 0 else np.inf,
            "photon_energy_eV": float(res["photon_energy"]["eV"]),
            "power_W": float(res["P_larmor"]),
            "relative_linewidth": float(res["relative_linewidth"]),
            "H3_over_H1": float(res["harmonic_ratios"]["H3_over_H1"]),
            "H5_over_H1": float(res["harmonic_ratios"]["H5_over_H1"]),
            "P_circ": float(res["Stokes"]["P_circ"]),
        })
    return results


def period_number_scan(
    gamma0,
    N_values=(10, 25, 50, 100, 200),
    r_obs=np.array([0.0, 0.0, 100.0]),
    realistic=True,
):
    """Full radiation scan versus number of undulator periods N_u."""
    results = []
    for N in N_values:
        und = make_default_undulator(realistic=realistic)
        und.device_n_periods = int(N)
        und.device_length = N * und.lambda_u
        K = und.K(me)
        t_span = simulation_span_for_device(
            gamma0, und, n_periods=N
        )
        n_base = samples_for_periods(N, pts_per_period=96)
        res = run_sim(
            und,
            beta_from_gamma(gamma0) * c0,
            t_span,
            np.asarray(r_obs, dtype=float),
            n_base=n_base,
            gamma0_input=gamma0,
        )
        results.append({
            "N_u": int(N),
            "f0_Hz": float(res["f0"]),
            "relative_linewidth": float(res["relative_linewidth"]),
            "fwhm_Hz": float(res["spectral_fwhm_hz"]),
            "quality_factor": float(res["spectral_quality_factor"]),
            "power_W": float(res["P_larmor"]),
            "equivalent_photons": float(res["photon_yield"]["equivalent_photons"]),
        })
    return results


def observation_distance_scan(
    gamma0,
    distances_m=(10.0, 30.0, 100.0, 300.0),
    n_periods=50,
    realistic=True,
):
    """Repeat the detector calculation at different on-axis observer distances.

    For far-field convergence, compare normalized spectral/angular quantities,
    not raw electric-field amplitude alone.
    """
    results = []
    und = make_default_undulator(realistic=realistic)
    und.device_n_periods = int(n_periods)
    und.device_length = n_periods * und.lambda_u
    K = und.K(me)
    t_span = simulation_span_for_device(
        gamma0, und, n_periods=n_periods
    )
    n_base = samples_for_periods(n_periods, pts_per_period=96)

    for R in distances_m:
        r_obs = np.array([0.0, 0.0, float(R)])
        res = run_sim(
            und,
            beta_from_gamma(gamma0) * c0,
            t_span,
            r_obs,
            n_base=n_base,
            gamma0_input=gamma0,
        )
        results.append({
            "distance_m": float(R),
            "f0_Hz": float(res["f0"]),
            "relative_linewidth": float(res["relative_linewidth"]),
            "P_circ": float(res["Stokes"]["P_circ"]),
            "fft_peak": float(np.max(res["fft"])) if len(res["fft"]) else 0.0,
            "equivalent_photon_rate_s^-1":
                float(res["photon_yield"]["equivalent_photon_rate_s^-1"]),
        })
    return results


def divergence_from_angle_scan(angle_result):
    """Convenience wrapper for the existing angle_scan output.

    angle_scan returns an ndarray with columns:
        theta, f0, fluence_J_m2, P_circ, P_lin
    Dict input is also accepted for forward compatibility.
    """
    if isinstance(angle_result, dict) and "divergence" in angle_result:
        return angle_result["divergence"]
    if isinstance(angle_result, dict) and "theta" in angle_result and "fluence_J_m2" in angle_result:
        return angular_divergence(angle_result["theta"], angle_result["fluence_J_m2"])
    if isinstance(angle_result, dict) and "theta" in angle_result and "Ipk" in angle_result:
        return angular_divergence(angle_result["theta"], angle_result["Ipk"])

    arr = np.asarray(angle_result)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return angular_divergence(arr[:, 0], arr[:, 2])

    raise ValueError(
        "angle_scan result must be an (N,>=3) array or contain theta and fluence_J_m2."
    )


def compare_device_presets(
    gamma0=100.0,
    preset_names=("planar", "helical", "elliptical", "left_helical", "wiggler"),
    n_periods=50,
    r_obs=np.array([0.0, 0.0, 100.0]),
    realistic=True,
):
    """Compare several insertion-device presets at the same electron gamma.

    This does NOT run automatically. It is kept separate so the default program
    still performs the original helical gamma/velocity scan.
    """
    rows = []
    for preset in preset_names:
        und = make_default_undulator(realistic=realistic, preset=preset)
        und.device_n_periods = int(n_periods)
        und.device_length = n_periods * und.lambda_u
        K = und.K(me)

        if ideal_beta_z_device(gamma0, und) <= 0.0:
            rows.append({
                "preset": preset,
                "valid": False,
                "reason": "gamma too low for nominal K",
            })
            continue

        t_span = simulation_span_for_device(
            gamma0, und, n_periods=n_periods
        )
        n_base = samples_for_periods(n_periods, pts_per_period=96)

        res = run_sim(
            und,
            beta_from_gamma(gamma0) * c0,
            t_span,
            np.asarray(r_obs, dtype=float),
            n_base=n_base,
            gamma0_input=gamma0,
        )

        rows.append({
            "preset": preset,
            "valid": True,
            "K_nominal": float(K),
            "Kx": float(res["K_components"]["Kx"]),
            "Ky": float(res["K_components"]["Ky"]),
            "photon_energy_eV": float(res["photon_energy"]["eV"]),
            "power_W": float(res["P_larmor"]),
            "P_linear": float(res["Stokes"]["P_lin"]),
            "P_circular": float(res["Stokes"]["P_circ"]),
            "relative_linewidth": float(res["relative_linewidth"]),
            "H3_over_H1": float(res["harmonic_ratios"]["H3_over_H1"]),
            "H5_over_H1": float(res["harmonic_ratios"]["H5_over_H1"]),
        })
    return rows



def run_scan_preset(
    scan_preset=SCAN_PRESET,
    device_preset=DEVICE_PRESET,
    realistic=APPLY_REALISTIC_ERRORS,
):
    """Unified scan entry point.

    One scan variable changes at a time; core analyses remain enabled.
    The default is still the original helical gamma scan.
    """
    if scan_preset not in SCAN_PRESETS:
        raise ValueError("Unknown SCAN_PRESET: %s" % scan_preset)

    cfg = SCAN_PRESETS[scan_preset]

    if scan_preset == "K":
        return k_scan(
            gamma0=cfg["gamma0"],
            K_values=cfg["K_values"],
            realistic=realistic,
        )

    if scan_preset == "N_periods":
        return period_number_scan(
            gamma0=cfg["gamma0"],
            N_values=cfg["N_values"],
            realistic=realistic,
        )

    if scan_preset == "observer_distance":
        return observation_distance_scan(
            gamma0=cfg["gamma0"],
            distances_m=cfg["distances_m"],
            realistic=realistic,
        )

    if scan_preset == "angle":
        und = make_default_undulator(
            realistic=realistic,
            preset=device_preset,
        )
        gamma0 = cfg["gamma0"]
        und.device_n_periods = 100
        und.device_length = 100 * und.lambda_u
        t_span = simulation_span_for_device(gamma0, und, n_periods=100)
        res = run_sim(
            und,
            beta_from_gamma(gamma0) * c0,
            t_span,
            np.array([0.0, 0.0, 100.0]),
            n_base=samples_for_periods(100),
            gamma0_input=gamma0,
        )
        theta = np.linspace(-5.0/gamma0, 5.0/gamma0, 81)
        ang = angle_scan(res, theta_range=theta)
        return {
            "angle_scan": ang,
            "divergence": divergence_from_angle_scan(ang),
        }

    # gamma is intentionally handled by the legacy/default parallel main path
    # so the original study, plots, and representative cases remain intact.
    return None


def _worker(args):
    gamma0, n_periods, r_obs = args
    und = make_default_undulator(
        preset=DEVICE_PRESET,
        field_model=FIELD_MODEL,
        n_periods=n_periods,
    )
    und.device_n_periods = int(n_periods)
    und.device_length = und.device_n_periods * und.lambda_u
    try:
        t_span = simulation_span_for_device(gamma0, und, n_periods=n_periods)
        n_base = samples_for_periods(n_periods)
        out = run_sim_scalar(und, None, t_span, r_obs, n_base=n_base, gamma0_input=gamma0)
        if out is not None:
            out["n_undulator_periods"] = int(n_periods)
            out["t_total"] = t_span[1] - t_span[0]
        return out
    except Exception as exc:
        print("scan point gamma=%.6g failed: %s" % (gamma0, exc))
        return None

if __name__ == "__main__":
    print(f"Field model: {FIELD_MODEL}")
    print(f"Device preset: {DEVICE_PRESET}")
    print(f"Scan preset: {SCAN_PRESET}")
    print(f"Analysis preset: {ANALYSIS_PRESET}")
    print("Error model: single realistic default")
    print(f"Quantum correction: {QUANTUM_CORRECTION} (monitor remains available)")
    print(f"Radiation reaction: {RADIATION_REACTION}")
    print(f"Fringe fields default: OFF")
    print("Available device presets:", ", ".join(list_device_presets()))
    print("Default workflow remains the original helical gamma/velocity scan.")
    if SCAN_PRESET != "gamma":
        scan_output = run_scan_preset(SCAN_PRESET, DEVICE_PRESET)
        print("[Selected scan output]")
        print(scan_output)
        raise SystemExit(0)
    B0 = 0.15
    lambda_u = 0.05
    N_PERIODS = 100
    und = make_default_undulator(
        preset=DEVICE_PRESET, field_model=FIELD_MODEL, n_periods=N_PERIODS
    )
    # Keep B0/lambda_u aliases synchronized with the configured model.
    B0 = und.B0
    lambda_u = und.lambda_u
    Ke = und.K(me)
    ku = und.k_u
    print("[Magnetic field model] %s" % FIELD_MODEL)
    print(und.error_summary())

    # Compare all energies over the SAME physical undulator length.
    # Each electron traverses exactly N_PERIODS magnetic periods; its own vz
    # determines the corresponding source-frame simulation time automatically.
    # Keep the finite magnetic device length synchronized with the simulated length.
    und.device_n_periods = N_PERIODS
    und.device_length = N_PERIODS * und.lambda_u
    ro = np.array([0.0, 0.0, 100.0])

    gamma_scan = get_adaptive_gamma_grid(gamma_min=1.25, gamma_max=6.0e4)
    v_scan_c = np.array([beta_from_gamma(g) for g in gamma_scan])
    v_scan = v_scan_c * c0
    worker_args = [(g, N_PERIODS, ro) for g in gamma_scan]

    t_scan0 = time.time()
    if FIELD_MODEL == "analytic":
        with ProcessPoolExecutor(max_workers=8) as executor:
            scan_raw = list(executor.map(_worker, worker_args))
    else:
        print("RADIA/field-map scan: sequential mode to avoid rebuilding/copying large maps across processes.")
        scan_raw = [_worker(a) for a in worker_args]
    scan_results = [r for r in scan_raw if r is not None]
    t_scan1 = time.time()
    print("Ultra-relativistic scan: %d/%d valid in %.1f s" % (len(scan_results), len(gamma_scan), t_scan1 - t_scan0))

    gs = np.array([r["gamma0"] for r in scan_results])
    vzs = np.array([r["v_z_avg"] / c0 for r in scan_results])
    vperps = Ke / gs
    rs_sim = np.array([r["R_avg"] * 1e6 for r in scan_results])
    lams = np.array([r["lambda0"] * 1e6 for r in scan_results])
    psim = np.array([r["P_larmor"] for r in scan_results])
    pth = np.array([r["P_schwinger"] for r in scan_results])
    fs = np.array([r["f0"] / 1e9 for r in scan_results])
    fths = np.array([r["f_expected"] / 1e9 for r in scan_results])
    photon_e = np.array([r["photon_energy_eV"] for r in scan_results])
    linewidths = np.array([r["relative_linewidth"] for r in scan_results])
    pcircs = np.array([r["P_circ"] for r in scan_results])
    plins = np.array([r["P_lin"] for r in scan_results])

    rep_gammas = np.array([1.25, 3.202563, 100.0, 6.0e4])
    rep_idx = np.array([int(np.argmin(np.abs(gs - g))) for g in rep_gammas])

    labs = [r"$\gamma$=%.3g ($\beta$=%.9f)" % (g, beta_from_gamma(g)) for g in rep_gammas]

    all_res = []
    for idx, g in enumerate(rep_gammas):
        t0 = time.time()
        t_span_g = simulation_span_for_device(g, und, n_periods=N_PERIODS)
        n_base_g = samples_for_periods(N_PERIODS)
        res = run_sim(und, None, t_span_g, ro, n_base=n_base_g, gamma0_input=g)
        res["n_undulator_periods"] = N_PERIODS
        res["t_total"] = t_span_g[1] - t_span_g[0]
        t1 = time.time()
        all_res.append(res)
        print("  gamma=%.6g: %d periods, t_total=%.6e s, done in %.1fs" %
              (g, N_PERIODS, res["t_total"], t1 - t0))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    print("\n[Undulator Parameters]")

    print("\n[Four Representative Cases]")
    header = (
        "%-8s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-12s %-10s %-10s"
        % ("v0", "gamma0", "gamma_avg", "vz_avg/c", "f_sim_GHz",
           "f_ideal_GHz", "lambda_um", "P_sim_W", "P_ideal_W",
           "P_circ", "P_lin")
    )
    print(header)
    print("-" * len(header))

    for lab, res in zip(labs, all_res):
        print(
            "%-8s %-10.6f %-10.6f %-12.8f %-12.6f %-12.6f %-12.6f %-12.6e %-12.6e %-10.6f %-10.6f"
            % (
                lab,
                res["gamma0"],
                res["gamma_avg"],
                res["v_z_avg"] / c0,
                res["f0"] / 1e9,
                res["f_expected"] / 1e9,
                res["lambda0"] * 1e6,
                res["P_larmor"],
                res["P_schwinger"],
                res["Stokes"]["P_circ"],
                res["Stokes"]["P_lin"],
            )
        )

        print(
            "    R_avg = %.6e m, R_exact = %.6e m, circularity = %.6f, "
            "pitch = %.6e m, peak_E = %.6e, FWHM = %.6e s, repetition = %.6e Hz"
            % (
                res["traj"]["avg_radius"],
                res["R_exact"],
                res["traj"]["circularity"],
                res["traj"]["pitch"],
                res["pulse"]["peak_amplitude"],
                res["pulse"]["avg_fwhm"],
                res["pulse"]["repetition_freq"],
            )
        )
        print(
            "    spectrum_FWHM = %.6e Hz, rel_linewidth = %.6e, Q = %.6e, "
            "energy_mismatch = %.3e, chi_max = %.3e, g_min = %.9f, aperture_lost = %s"
            % (
                res["spectral_fwhm_hz"],
                res["relative_linewidth"],
                res["spectral_quality_factor"],
                res["energy_accounting"]["relative_mismatch"],
                res["quantum"]["chi_max"],
                res["quantum"]["g_min"],
                res["lost_to_aperture"],
            )
        )
        print(
            "    photon_E = %.6e eV (%.6e keV), N_gamma(eq) = %.6e, "
            "photon_rate(eq) = %.6e s^-1, H3/H1 = %.6e, H5/H1 = %.6e"
            % (
                res["photon_energy"]["eV"],
                res["photon_energy"]["keV"],
                res["photon_yield"]["equivalent_photons"],
                res["photon_yield"]["equivalent_photon_rate_s^-1"],
                res["harmonic_ratios"]["H3_over_H1"],
                res["harmonic_ratios"]["H5_over_H1"],
            )
        )

    print("\n[Ultra-Relativistic Scan Summary]")

    print("\n[Ultra-Relativistic Scan: Full Per-Point Data]")
    hdr_a = (
        "%-5s %-8s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-12s %-13s %-13s"
        % ("idx", "v0/c", "gamma0", "gamma_avg", "vz/c", "vz_avg/c",
           "f0_GHz", "f_ideal_GHz", "lambda0_um", "lam_ideal_um",
           "P_larmor_W", "P_schwinger_W")
    )
    print(hdr_a)
    print("-" * len(hdr_a))
    for i, r in enumerate(scan_results):
        print(
            "%-5d %-8.4f %-10.6f %-10.6f %-12.8f %-12.8f %-12.6f %-12.6f %-12.6f %-12.6f %-13.6e %-13.6e"
            % (
                i,
                r["v0"] / c0,
                r["gamma0"],
                r["gamma_avg"],
                r["v_z"] / c0,
                r["v_z_avg"] / c0,
                r["f0"] / 1e9,
                r["f_expected"] / 1e9,
                r["lambda0"] * 1e6,
                r["lam_theory"] * 1e6,
                r["P_larmor"],
                r["P_schwinger"],
            )
        )

    hdr_b = (
        "%-5s %-8s %-12s %-12s %-12s %-12s %-12s %-13s %-12s %-12s %-10s %-10s %-12s %-10s %-12s %-11s"
        % ("idx", "v0/c", "R_avg_um", "R_max_um", "R_exact_um", "circularity",
           "pitch_m", "peak_E", "FWHM_s", "rep_Hz", "P_circ", "P_lin",
           "MLE", "mle_count", "r_diffusion", "n_poincare")
    )
    print(hdr_b)
    print("-" * len(hdr_b))
    for i, r in enumerate(scan_results):
        print(
            "%-5d %-8.4f %-12.6f %-12.6f %-12.6f %-12.6f %-12.6e %-13.6e %-12.6e %-12.6e %-10.6f %-10.6f %-12.6e %-10d %-12.6f %-11d"
            % (
                i,
                r["v0"] / c0,
                r["R_avg"] * 1e6,
                r["R_max"] * 1e6,
                r["R_exact"] * 1e6,
                r["circularity"],
                r["pitch"],
                r["peak_amplitude"],
                r["avg_fwhm"],
                r["repetition_freq"],
                r["P_circ"],
                r["P_lin"],
                r["MLE"],
                r["mle_count"],
                r["r_diffusion"],
                r["n_poincare"],
            )
        )

    gth = np.geomspace(np.sqrt(1 + Ke ** 2) + 0.01, 6.0e4, 700)
    vperp_ideal = Ke / gth
    vz_ideal = np.sqrt(1.0 - (1.0 + Ke ** 2) / gth ** 2)
    delta_ideal = (1.0 + Ke ** 2) / (gth ** 2 * (1.0 + vz_ideal))
    lam_ideal = lambda_u * delta_ideal / vz_ideal * 1e6

    fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10), dpi=110)

    ax = axs1[0, 0]
    ax.plot(gs, vzs, '-', color='#1f77b4', lw=2, alpha=0.8, label=r'Sim $v_z/c$')
    ax.plot(gs, vperps, '-', color='#ff7f0e', lw=2, alpha=0.8, label=r'Sim $v_\perp/c$')
    ax.plot(gth, vz_ideal, '--', color='black', lw=1.5, alpha=0.6, label=r'Ideal $v_z/c$')
    ax.plot(gth, vperp_ideal, '--', color='gray', lw=1.5, alpha=0.6, label=r'Ideal $v_\perp/c$')
    ax.scatter(gs[rep_idx], vzs[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.scatter(gs[rep_idx], vperps[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel('Velocity / c', fontweight='bold')
    ax.set_title('(a) Velocity Decomposition', fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls='--', alpha=0.5)

    ax = axs1[0, 1]
    ax.plot(gs, lams, '-', color='#1f77b4', lw=2, alpha=0.8, label=r'Sim $\lambda$')
    ax.plot(gth, lam_ideal, '--', color='black', lw=1.5, alpha=0.6, label=r'Ideal $\lambda=\lambda_u(c/v_z-1)$')
    ax.scatter(gs[rep_idx], lams[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\lambda$ ($\mu$m)', fontweight='bold')
    ax.set_title('(b) Wavelength', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    ax = axs1[0, 2]
    ax.plot(gs, psim, '-', color='#1f77b4', lw=2, alpha=0.8, label='Sim')
    ax.plot(gs, pth, '--', color='black', lw=1.5, alpha=0.6, label='Ideal analytical power')
    ax.scatter(gs[rep_idx], psim[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel(r'$P$ (W)', fontweight='bold')
    ax.set_title('(c) Radiated Power', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(fontsize=9)

    ax = axs1[1, 0]
    ax.plot(gs, rs_sim, '-', color='#1f77b4', lw=2, alpha=0.8, label=r'Sim $R_{\rm avg}$')
    ax.scatter(gs[rep_idx], rs_sim[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    gth_r = np.geomspace(np.sqrt(1 + Ke ** 2) + 0.01, 6.0e4, 700)
    Rex_ref = []
    for g in gth_r:
        vp = Ke * c0 / g
        v0 = beta_from_gamma(g) * c0
        vz = np.power(max(v0 ** 2 - vp ** 2, 0), 0.5)
        Rex_ref.append(vp / (ku * vz) * 1e6 if vz > 0 else np.nan)
    ax.plot(gth_r, Rex_ref, '--', color='black', lw=1.5, alpha=0.6, label=r'Ideal $R_{\rm ex}$')
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel(r'$R_{\rm avg}$ ($\mu$m)', fontweight='bold')
    ax.set_title('(d) Average Radius', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    ax = axs1[1, 1]
    ax.plot(gs, photon_e, '-', color='#d62728', lw=2, alpha=0.8, label='Photon energy')
    ax.scatter(gs[rep_idx], photon_e[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel(r'$E_\gamma$ (eV)', fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('(e) Photon Energy', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls='--', alpha=0.5)

    ax = axs1[1, 2]
    ax.plot(gs, pcircs, '-', color='#9467bd', lw=2, alpha=0.8, label=r'Sim $P_{\rm circ}$')
    ax.plot(gs, plins, '-', color='#ff7f0e', lw=2, alpha=0.8, label=r'Sim $P_{\rm lin}$')
    ax.axhline(-1.0, color='black', ls='--', lw=1.5, alpha=0.6, label=r'Ideal $P_{\rm circ}=-1$')
    ax.axhline(0.0, color='gray', ls='--', lw=1.5, alpha=0.6, label=r'Ideal $P_{\rm lin}=0$')
    ax.scatter(gs[rep_idx], pcircs[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.scatter(gs[rep_idx], plins[rep_idx], s=80, facecolors='none', edgecolors='red', linewidths=2, zorder=5)
    ax.set_xlabel(r'$\gamma$', fontweight='bold')
    ax.set_xscale('log')
    ax.set_ylabel('Polarization Degree', fontweight='bold')
    ax.set_title('(f) On-axis Polarization', fontweight='bold')
    ax.set_ylim(-1.05, 0.2)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls='--', alpha=0.5)

    fig1.suptitle(r'Parametric Dependence on $\gamma$ (K=%.2f, ultra-relativistic gamma scan)' % Ke, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    import matplotlib.gridspec as gridspec

    fig2 = plt.figure(figsize=(18, 10), dpi=110)
    gsp = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.35, wspace=0.3)

    for idx, (res, lab) in enumerate(zip(all_res, labs)):
        if idx == 0:
            ax = fig2.add_subplot(gsp[0, 0])
        elif idx == 1:
            ax = fig2.add_subplot(gsp[0, 1])
        elif idx == 2:
            ax = fig2.add_subplot(gsp[0, 2])
        else:
            ax = fig2.add_subplot(gsp[1, 0])

        fghz = res["freq"] / 1e9
        amp = res["fft"]
        amp_norm = amp / np.max(amp) if np.max(amp) > 1e-30 else amp

        ax.plot(fghz, amp_norm, color=colors[idx], lw=1.5, label=lab)
        ax.axvline(res["f0"] / 1e9, color=colors[idx], ls='--', lw=1.5, alpha=0.8,
                   label=r'$f_{\rm sim}$')
        ax.axvline(res["f_expected"] / 1e9, color='black', ls=':', lw=1.5, alpha=0.6,
                   label=r'$f_{\rm ideal}$')

        f0_ghz = res["f0"] / 1e9
        x_max = max(3.0, 2.5 * f0_ghz)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Frequency (GHz)', fontweight='bold')
        ax.set_ylabel('Normalized FFT', fontweight='bold')
        ax.set_title(r'(%s) %s, $f_0$=%.2f GHz' % (chr(97 + idx), lab, f0_ghz),
                     fontweight='bold')

        ax.text(0.97, 0.95,
                r'$f_{\rm sim}=%.3f$ GHz' % (res["f0"] / 1e9) + '\n' +
                r'$f_{\rm exact}=%.3f$ GHz' % (res["f_expected"] / 1e9),
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, ls='--', alpha=0.5)

    ax_b = fig2.add_subplot(gsp[1, 1:3])
    ax_b.plot(gs, fs, '-', color='#1f77b4', lw=1.5, alpha=0.8, label='Sim')
    ax_b.plot(gs, fths, '--', color='black', lw=1.5, alpha=0.6,
              label=r'Ideal helical reference')
    ax_b.scatter(gs[rep_idx], fs[rep_idx], s=80, facecolors='none',
                 edgecolors='red', linewidths=2, zorder=5)
    ax_b.set_xlabel(r'$\gamma$', fontweight='bold')
    ax_b.set_xscale('log')
    ax_b.set_ylabel(r'$f_0$ (GHz)', fontweight='bold')
    ax_b.set_title('(e) Frequency vs Lorentz Factor', fontweight='bold')
    ax_b.set_yscale('log')
    ax_b.grid(True, which='both', ls='--', alpha=0.5)
    ax_b.legend(fontsize=9)

    fig2.suptitle('Spectral Validation vs Ideal Helical Reference', fontsize=14, fontweight='bold', y=0.98)
    plt.show()
    from scipy.special import jv as besselj

    chaos_results = []
    for idx, res in enumerate(all_res):
        ch = chaos_analysis(res, und)
        chaos_results.append(ch)

    if 'scan_results' in globals() and len(scan_results) > 0:
        gs = np.array([r["gamma0"] for r in scan_results])
        r_diffs_scan = np.array([r["r_diffusion"] for r in scan_results])
        circs_scan = np.array([r["circularity"] for r in scan_results])
        mles_scan = np.array([r["MLE"] for r in scan_results])
        n_poincs = np.array([r["n_poincare"] for r in scan_results])
        mle_counts = np.array([r["mle_count"] for r in scan_results])
        has_50 = True
    else:
        gs = np.array([r["gamma0"] for r in all_res])
        r_diffs_scan = np.array([ch["r_diffusion"] for ch in chaos_results])
        circs_scan = np.array([ch["circularity"] for ch in chaos_results])
        mles_scan = np.array([ch["mle"] for ch in chaos_results])
        n_poincs = np.array([ch["n_poincare"] for ch in chaos_results])
        mle_counts = np.array([ch["mle_count"] for ch in chaos_results])
        has_50 = False

    r_diff_mean = np.mean(r_diffs_scan) if len(r_diffs_scan) > 0 else 0.56
    r_diff_std = np.std(r_diffs_scan) if len(r_diffs_scan) > 0 else 0.0

    circ_mean = np.mean(circs_scan) if len(circs_scan) > 0 else 0.05
    circ_std = np.std(circs_scan) if len(circs_scan) > 0 else 0.0

    fig3 = plt.figure(figsize=(18, 10))

    ax_3d_0 = fig3.add_subplot(2, 2, 1, projection='3d')
    res = all_res[0]
    x, y, z = res["r"][:, 0], res["r"][:, 1], res["r"][:, 2]
    n_periods = 20
    z_max = z[0] + n_periods * und.lambda_u
    mask = z <= z_max
    ax_3d_0.plot(x[mask] * 1e6, y[mask] * 1e6, z[mask] * 1e3, color=colors[0], lw=1.5)
    ax_3d_0.set_xlabel(r"X ($\mu$m)", fontweight='bold')
    ax_3d_0.set_ylabel(r"Y ($\mu$m)", fontweight='bold')
    ax_3d_0.set_zlabel("Z (mm)", fontweight='bold')
    ax_3d_0.set_title(r"(a) $\gamma$=%.3g, $\beta$=%.9f" % (all_res[0]['gamma0'], beta_from_gamma(all_res[0]['gamma0'])), fontsize=10, fontweight='bold')

    ax_3d_1 = fig3.add_subplot(2, 2, 2, projection='3d')
    res = all_res[3]
    x, y, z = res["r"][:, 0], res["r"][:, 1], res["r"][:, 2]
    z_max = z[0] + n_periods * und.lambda_u
    mask = z <= z_max
    ax_3d_1.plot(x[mask] * 1e6, y[mask] * 1e6, z[mask] * 1e3, color=colors[3], lw=1.5)
    ax_3d_1.set_xlabel(r"X ($\mu$m)", fontweight='bold')
    ax_3d_1.set_ylabel(r"Y ($\mu$m)", fontweight='bold')
    ax_3d_1.set_zlabel("Z (mm)", fontweight='bold')
    ax_3d_1.set_title(r"(b) $\gamma$=%.3g, $\beta$=%.9f" % (all_res[3]['gamma0'], beta_from_gamma(all_res[3]['gamma0'])), fontsize=10, fontweight='bold')

    ax_rdiff = fig3.add_subplot(2, 2, 3)
    ax_rdiff.plot(gs, r_diffs_scan, 'o', color='#ff7f0e', ms=4, alpha=0.6,
                  label=r'Sim $r_{\rm diff}$ (%d pts)' % len(r_diffs_scan))
    if has_50:
        rep_idx = np.array([int(np.argmin(np.abs(gs - g))) for g in rep_gammas])
        rep_idx = np.array([i for i in rep_idx if i < len(gs)])
        if len(rep_idx) > 0:
            ax_rdiff.scatter(gs[rep_idx], r_diffs_scan[rep_idx], s=80, facecolors='none',
                             edgecolors='red', linewidths=2, zorder=5, label='Rep. cases')

    if len(gs) >= 4 and len(r_diffs_scan) == len(gs):
        gs_sorted_idx = np.argsort(gs)
        gs_sorted = gs[gs_sorted_idx]
        r_diff_sorted = r_diffs_scan[gs_sorted_idx]
        _, unique_idx = np.unique(gs_sorted, return_index=True)
        gs_unique = gs_sorted[unique_idx]
        r_diff_unique = r_diff_sorted[unique_idx]
        if len(gs_unique) >= 4:
            cs_rdiff = CubicSpline(gs_unique, r_diff_unique)
            gs_fine = np.linspace(gs_unique.min(), gs_unique.max(), 300)
            ax_rdiff.plot(gs_fine, cs_rdiff(gs_fine), '-', color='#d62728', lw=2.5, alpha=0.9,
                          label=r'CubicSpline $r_{\rm diff}$')

    ax_rdiff.axhline(0.0, color='black', ls='--', lw=1.5, alpha=0.6,
                     label=r'Ideal ($r_{\rm diff}=0$)')
    ax_rdiff.set_xlabel(r'$\gamma$', fontweight='bold')
    ax_rdiff.set_xscale('log')
    ax_rdiff.set_ylabel(r'$r_{\rm diff}$', fontweight='bold')
    ax_rdiff.set_title('(c) Radial Diffusion', fontweight='bold')
    ax_rdiff.legend(fontsize=8, loc='upper right')
    ax_rdiff.grid(True, ls='--', alpha=0.5)

    ax_circ = fig3.add_subplot(2, 2, 4)
    ax_circ.plot(gs, circs_scan, 'o', color='#2ca02c', ms=4, alpha=0.6,
                 label=r'Sim $\mathcal{C}$ (%d pts)' % len(circs_scan))
    if has_50 and len(rep_idx) > 0:
        ax_circ.scatter(gs[rep_idx], circs_scan[rep_idx], s=80, facecolors='none',
                        edgecolors='red', linewidths=2, zorder=5, label='Rep. cases')

    if len(gs) >= 4 and len(circs_scan) == len(gs):
        gs_sorted_idx = np.argsort(gs)
        gs_sorted = gs[gs_sorted_idx]
        circ_sorted = circs_scan[gs_sorted_idx]
        _, unique_idx = np.unique(gs_sorted, return_index=True)
        gs_unique = gs_sorted[unique_idx]
        circ_unique = circ_sorted[unique_idx]
        if len(gs_unique) >= 4:
            cs_circ = CubicSpline(gs_unique, circ_unique)
            gs_fine = np.linspace(gs_unique.min(), gs_unique.max(), 300)
            ax_circ.plot(gs_fine, cs_circ(gs_fine), '-', color='#9467bd', lw=2.5, alpha=0.9,
                         label=r'CubicSpline $\mathcal{C}$')

    ax_circ.axhline(1.0, color='black', ls='--', lw=1.5, alpha=0.6,
                    label=r'Ideal ($\mathcal{C}=1$)')
    ax_circ.set_xlabel(r'$\gamma$', fontweight='bold')
    ax_circ.set_xscale('log')
    ax_circ.set_ylabel('Circularity', fontweight='bold')
    ax_circ.set_title('(d) Orbit Circularity', fontweight='bold')
    ax_circ.set_ylim(0, 1.1)
    ax_circ.legend(fontsize=8, loc='upper right')
    ax_circ.grid(True, ls='--', alpha=0.5)
    fig3.suptitle('Nonlinear Dynamics & Phase Space Diagnostics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Adaptive off-axis scan: resolve the relativistic radiation cone using u = gamma*theta.
    # A fixed milliradian grid badly undersamples ultra-relativistic cases.
    OFFAXIS_U_MAX = 5.0
    OFFAXIS_N_THETA = 81
    offaxis_results = []
    for idx, res in enumerate(all_res):
        gamma_ang = max(float(res["gamma_avg"]), 1.0)
        theta_range = np.linspace(-OFFAXIS_U_MAX / gamma_ang,
                                  OFFAXIS_U_MAX / gamma_ang,
                                  OFFAXIS_N_THETA)
        sc = angle_scan(res, theta_range, n_obs=4000)
        I_max = np.max(sc[:, 2]) if np.max(sc[:, 2]) > 0 else 1.0
        mask_weak = sc[:, 2] < 1e-3 * I_max
        sc[mask_weak, 3] = np.nan
        sc[mask_weak, 4] = np.nan
        offaxis_results.append(sc)

    print("\n[Off-Axis Scan Parameters]")

    print("\n[Off-Axis Data for Four Representative Cases]")
    for idx, (sc, lab) in enumerate(zip(offaxis_results, labs)):
        print("%-10s %-12s %-14s %-14s %-14s %-14s" % ("theta(mrad)", "gamma*theta", "f0(GHz)", "Fluence(J/m2)", "P_circ", "P_lin"))
        print("-" * 84)
        for row in sc:
            theta_mrad = row[0] * 1e3
            gamma_theta = float(res["gamma_avg"]) * row[0]
            f0_ghz = row[1] / 1e9 if row[1] > 0 else 0.0
            ipk = row[2]
            pcirc = row[3] if not np.isnan(row[3]) else 0.0
            plin = row[4] if not np.isnan(row[4]) else 0.0
            print("%-10.4f %-12.5f %-14.6f %-14.6e %-14.6f %-14.6f" %
                  (theta_mrad, gamma_theta, f0_ghz, ipk, pcirc, plin))

        valid_pcirc = sc[~np.isnan(sc[:, 3])]
        if len(valid_pcirc) > 0:
            print("  P_circ range: %.6f to %.6f" % (np.min(valid_pcirc[:, 3]), np.max(valid_pcirc[:, 3])))
        valid_plin = sc[~np.isnan(sc[:, 4])]
        if len(valid_plin) > 0:
            print("  P_lin range: %.6f to %.6f" % (np.min(valid_plin[:, 4]), np.max(valid_plin[:, 4])))
        valid_f = sc[sc[:, 1] > 0]
        if len(valid_f) > 0:
            print("  f0 range: %.6f to %.6f GHz" % (np.min(valid_f[:, 1])/1e9, np.max(valid_f[:, 1])/1e9))
        print("  Fluence max (J/m^2): %.6e" % np.nanmax(sc[:, 2]))

    fig4, axs4 = plt.subplots(1, 3, figsize=(18, 5), dpi=100)

    ax = axs4[0]
    for idx, (sc, lab, res) in enumerate(zip(offaxis_results, labs, all_res)):
        ok = sc[:, 1] > 0
        ax.plot(res['gamma_avg'] * sc[ok, 0], sc[ok, 1] / 1e9, 'o-', color=colors[idx], lw=2, ms=5, label=lab)
        ax.axhline(all_res[idx]['f0'] / 1e9, color=colors[idx], ls='--', lw=1, alpha=0.5)

        gamma = res['gamma_avg']
        K = res['K']
        f0 = res['f_expected']
        theta_f = np.linspace(-OFFAXIS_U_MAX / gamma, OFFAXIS_U_MAX / gamma, 300)
        denom = 1.0 + K ** 2 + (gamma * theta_f) ** 2
        f_ideal = f0 * (1.0 + K ** 2) / denom
        ax.plot(gamma * theta_f, f_ideal / 1e9, '--', color=colors[idx], lw=1.5, alpha=0.6)

    ax.set_xlabel(r'$\gamma\theta$', fontweight='bold')
    ax.set_ylabel(r'$f_0$ (GHz)', fontweight='bold')
    ax.set_title('(a) Frequency Redshift', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.5)

    ax = axs4[1]
    for idx, (sc, lab, res) in enumerate(zip(offaxis_results, labs, all_res)):
        I = sc[:, 2]
        In = I / I.max() if I.max() > 0 else I
        ok = I > 0
        ax.plot(res['gamma_avg'] * sc[ok, 0], In[ok], 'o-', color=colors[idx], lw=2, ms=5, label=lab)

        gamma = res['gamma_avg']
        K = res['K']
        theta_i = np.linspace(-OFFAXIS_U_MAX / gamma, OFFAXIS_U_MAX / gamma, 300)
        xi = K * gamma * np.abs(theta_i) / (1.0 + K ** 2)
        J0 = besselj(0, xi)
        J1 = besselj(1, xi)
        I_ref = (J0 ** 2 + J1 ** 2) / ((1.0 + (gamma * theta_i) ** 2 / (1.0 + K ** 2)) ** 2)
        I_ref = I_ref / I_ref.max()
        ax.plot(gamma * theta_i, I_ref, '--', color=colors[idx], lw=1.5, alpha=0.6)

    ax.set_xlabel(r'$\gamma\theta$', fontweight='bold')
    ax.set_ylabel(r'$I/I_{\max}$', fontweight='bold')
    ax.set_title('(b) Intensity Profile', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.5)

    ax = axs4[2]
    for idx, (sc, lab, res) in enumerate(zip(offaxis_results, labs, all_res)):
        ok = ~np.isnan(sc[:, 3])
        ax.plot(res['gamma_avg'] * sc[ok, 0], sc[ok, 3], 'o-', color=colors[idx], lw=2, ms=5, label=lab)

        gamma = res['gamma_avg']
        K = res['K']
        theta_p = np.linspace(-OFFAXIS_U_MAX / gamma, OFFAXIS_U_MAX / gamma, 300)
        xi = K * gamma * np.abs(theta_p) / (1.0 + K ** 2)
        J0 = besselj(0, xi)
        J1 = besselj(1, xi)
        denom = J0 ** 2 + J1 ** 2
        denom_safe = np.where(denom < 1e-30, 1e-30, denom)
        Pcirc_ref = -(J0 ** 2 - J1 ** 2) / denom_safe
        Pcirc_ref = np.clip(Pcirc_ref, -1.0, 1.0)
        ax.plot(gamma * theta_p, Pcirc_ref, '--', color=colors[idx], lw=1.5, alpha=0.6)

    ax.axhline(-1.0, color='gray', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\gamma\theta$', fontweight='bold')
    ax.set_ylabel(r'$P_{\rm circ}$', fontweight='bold')
    ax.set_title('(c) Polarization Degradation', fontweight='bold')
    ax.set_ylim(-1.05, 0.05)
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.5)

    fig4.suptitle(r'Adaptive Off-Axis Scan in $\gamma\theta$ (K=%.2f)' % Ke, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()