from __future__ import annotations
import io
import json
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go


PLOT_HEIGHT = 650
PLOT_HEIGHT_3D = 760
TABLE_HEIGHT = 620


def _arr(x):
    return np.asarray(x, dtype=float)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "splines"}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _flatten(prefix, value, rows):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, rows)
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            rows.append({"quantity": prefix, "value": value.item(), "unit_or_shape": ""})
        else:
            rows.append({"quantity": prefix, "value": "array", "unit_or_shape": str(tuple(value.shape))})
    elif isinstance(value, (list, tuple)) and len(value) > 8:
        rows.append({"quantity": prefix, "value": "sequence", "unit_or_shape": str((len(value),))})
    elif isinstance(value, (str, int, float, bool, np.generic)) or value is None:
        val = value.item() if isinstance(value, np.generic) else value
        rows.append({"quantity": prefix, "value": val, "unit_or_shape": ""})


def scalar_result_table(r):
    rows = []
    skip = {
        "r", "u", "t_src", "g_arr", "P_larmor_t", "t_obs", "E", "E_total",
        "E1", "E2", "freq", "fft", "tp_log", "splines"
    }
    for k, v in r.items():
        if k in skip:
            continue
        _flatten(k, v, rows)
    return pd.DataFrame(rows)


def _velocity_arrays(r, v11):
    u = _arr(r["u"])
    gam = _arr(r["g_arr"])
    return u / (gam[:, None] * float(v11.me))


def result_tables(r, v11, dev=None, field_quality_fn=None):
    rpos = _arr(r["r"])
    ts = _arr(r["t_src"])
    gam = _arr(r["g_arr"])
    P = _arr(r["P_larmor_t"])
    vel = _velocity_arrays(r, v11)
    vz = np.where(np.abs(vel[:, 2]) > 1e-30, vel[:, 2], np.nan)

    trajectory = pd.DataFrame({
        "t_src_s": ts,
        "x_m": rpos[:, 0],
        "y_m": rpos[:, 1],
        "z_m": rpos[:, 2],
        "gamma": gam,
        "vx_m_s": vel[:, 0],
        "vy_m_s": vel[:, 1],
        "vz_m_s": vel[:, 2],
        "xprime_rad": vel[:, 0] / vz,
        "yprime_rad": vel[:, 1] / vz,
        "P_larmor_W": P,
    })

    tobs = _arr(r["t_obs"])
    E = _arr(r["E"])
    observer = pd.DataFrame({
        "t_obs_s": tobs,
        "Ex_V_m": E[:, 0],
        "Ey_V_m": E[:, 1],
        "Ez_V_m": E[:, 2],
        "E1_V_m": _arr(r["E1"]),
        "E2_V_m": _arr(r["E2"]),
        "E_magnitude_V_m": np.linalg.norm(E, axis=1),
    })

    spectrum = pd.DataFrame({
        "frequency_Hz": _arr(r["freq"]),
        "fft_amplitude": _arr(r["fft"]),
    })

    q = r.get("quantum", {})
    quantum = None
    if isinstance(q, dict) and "chi_array" in q and len(np.asarray(q["chi_array"])) == len(ts):
        quantum = pd.DataFrame({
            "t_src_s": ts,
            "z_m": rpos[:, 2],
            "chi_e": _arr(q["chi_array"]),
            "gaunt_factor": _arr(q["g_array"]),
        })

    field = None
    fq = None
    if dev is not None:
        z0 = float(rpos[0, 2])
        z1 = float(rpos[-1, 2])
        z = np.linspace(z0, z1, max(501, min(5001, len(ts))))
        pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        try:
            B = _arr(dev.B(pts))
            field = pd.DataFrame({
                "z_m": z,
                "Bx_T": B[:, 0],
                "By_T": B[:, 1],
                "Bz_T": B[:, 2],
                "Bperp_T": np.hypot(B[:, 0], B[:, 1]),
            })
        except Exception:
            field = None
        if field_quality_fn is not None:
            try:
                nper = max(1, int(round((z1 - z0) / float(dev.lambda_u))))
                fq = field_quality_fn(dev, nper)
            except Exception:
                fq = None

    return {
        "trajectory": trajectory,
        "observer": observer,
        "spectrum": spectrum,
        "quantum": quantum,
        "field": field,
        "field_quality": fq,
        "scalars": scalar_result_table(r),
    }


def result_bundle_zip(r, v11, dev=None, field_quality_fn=None, extras=None):
    tables = result_tables(r, v11, dev, field_quality_fn)
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "README.txt",
            "V11 RADIA full numerical result export\n"
            "Arrays are exported as CSV where appropriate; nested diagnostics are in JSON.\n"
            "The radiation model is single-electron research software.\n",
        )
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                zf.writestr(f"{name}.csv", df.to_csv(index=False))
        if isinstance(tables.get("field_quality"), dict):
            zf.writestr(
                "field_quality.json",
                json.dumps(_json_safe(tables["field_quality"]), indent=2),
            )
        for name in [
            "Stokes", "trajectory_phase", "energy_accounting", "quantum",
            "photon_yield", "spectral_photon_yield", "theory_residuals",
            "harmonic_ratios", "pulse", "traj", "K_components"
        ]:
            if name in r:
                zf.writestr(f"{name}.json", json.dumps(_json_safe(r[name]), indent=2))
        zf.writestr("complete_result.json", json.dumps(_json_safe(r), indent=2))
        if extras:
            for name, obj in extras.items():
                if isinstance(obj, pd.DataFrame):
                    zf.writestr(f"extras/{name}.csv", obj.to_csv(index=False))
                elif isinstance(obj, dict):
                    zf.writestr(
                        f"extras/{name}.json",
                        json.dumps(_json_safe(obj), indent=2),
                    )
    return bio.getvalue()


def _fmt_value(v, scientific=False):
    try:
        x = float(v)
        if not np.isfinite(x):
            return str(x)
        return f"{x:.12e}" if scientific else f"{x:.12g}"
    except Exception:
        return str(v)


def _dict_df(obj):
    rows = []
    for k, v in dict(obj or {}).items():
        if isinstance(v, (int, float, np.number)):
            shown = _fmt_value(v, scientific=True)
        else:
            shown = str(v)
        rows.append({"quantity": str(k), "value": shown})
    return pd.DataFrame(rows)


def _numeric_column_config(st, df):
    cfg = {}
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(
                str(col),
                format="%.10e",
            )
        elif pd.api.types.is_integer_dtype(df[col]):
            cfg[col] = st.column_config.NumberColumn(
                str(col),
                format="%d",
            )
    return cfg


def _show_table(st, title, df, key, height=TABLE_HEIGHT, caption=None):
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No numerical table is available for this item.")
        return
    st.dataframe(
        df,
        width="stretch",
        height=int(height),
        hide_index=True,
        column_config=_numeric_column_config(st, df),
    )
    st.download_button(
        f"Download {title} CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_fullresults_{key}",
        use_container_width=True,
    )


def _trace(x, y, name, mode="lines"):
    return go.Scatter(
        x=x,
        y=y,
        mode=mode,
        name=name,
        hovertemplate=(
            f"{name}<br>x=%{{x:.10e}}<br>y=%{{y:.10e}}<extra></extra>"
        ),
    )


def _style(fig, title, x_title, y_title, height=PLOT_HEIGHT, hovermode="x unified"):
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=int(height),
        autosize=True,
        hovermode=hovermode,
        margin=dict(l=95, r=45, t=125, b=90),
        font=dict(size=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
            itemwidth=55,
        ),
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(
            automargin=True,
            tickformat=".7g",
            hoverformat=".10e",
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(
            automargin=True,
            tickformat=".7g",
            hoverformat=".10e",
            showspikes=True,
        ),
    )
    return fig


def _plot(st, fig, title, x_title, y_title, height=PLOT_HEIGHT, hovermode="x unified"):
    _style(fig, title, x_title, y_title, height, hovermode)
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": title.lower().replace(" ", "_").replace("/", "_"),
                "scale": 2,
            },
        },
    )


def _cumtrap(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x))
    if len(x) > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _summary_table(r, stokes):
    rows = [
        ("Photon energy", r["photon_energy"]["eV"], "eV"),
        ("Fundamental frequency", r["f0"], "Hz"),
        ("Theoretical fundamental frequency", r["f_expected"], "Hz"),
        ("Fundamental wavelength", r["lambda0"], "m"),
        ("Average radiated power", r["P_larmor"], "W"),
        ("Relative linewidth", r["relative_linewidth"], ""),
        ("Quality factor Q", r["spectral_quality_factor"], ""),
        ("Circular polarization", stokes.get("P_circ", np.nan), ""),
        ("Linear polarization", stokes.get("P_lin", np.nan), ""),
        ("Radiation H3/H1", r["harmonic_ratios"]["H3_over_H1"], ""),
        ("Radiation H5/H1", r["harmonic_ratios"]["H5_over_H1"], ""),
        ("Initial gamma", r["gamma0"], ""),
        ("Average gamma", r["gamma_avg"], ""),
        ("Observer angle", float(r.get("observer_theta_rad", 0.0)) * 1e3, "mrad"),
    ]
    return pd.DataFrame({
        "quantity": [x[0] for x in rows],
        "value": [_fmt_value(x[1], scientific=True) for x in rows],
        "unit": [x[2] for x in rows],
    })


def render_full_results(st, r, v11, dev=None, field_quality_fn=None):
    """Render a deliberately long, wide, scrollable report without re-running physics."""
    tables = result_tables(r, v11, dev, field_quality_fn)

    pos = _arr(r["r"])
    z = pos[:, 2]
    ts = _arr(r["t_src"])
    gam = _arr(r["g_arr"])
    P = _arr(r["P_larmor_t"])
    vel = _velocity_arrays(r, v11)
    beta = vel / float(v11.c0)
    vz = np.where(np.abs(vel[:, 2]) > 1e-30, vel[:, 2], np.nan)

    tobs = _arr(r["t_obs"])
    E = _arr(r["E"])
    E1 = _arr(r["E1"])
    E2 = _arr(r["E2"])
    freq = _arr(r["freq"])
    fft = _arr(r["fft"])
    tp = _arr(r.get("tp_log", np.full_like(tobs, np.nan)))
    stokes = dict(r.get("Stokes", {}))

    st.markdown("## Comprehensive result output — expanded visualization")
    st.caption(
        "Long-scroll mode: figures are intentionally full-width and tall. "
        "Hover values use high precision, major numerical tables are displayed directly, "
        "and every table can also be downloaded as CSV."
    )

    # Precise summary: only two cards per row to avoid clipping.
    summary_metrics = [
        ("Photon energy", f"{float(r['photon_energy']['eV']):.12e} eV"),
        ("Fundamental frequency", f"{float(r['f0']):.12e} Hz"),
        ("Fundamental wavelength", f"{float(r['lambda0']):.12e} m"),
        ("Average radiated power", f"{float(r['P_larmor']):.12e} W"),
        ("Relative linewidth", f"{float(r['relative_linewidth']):.12e}"),
        ("Quality factor Q", f"{float(r['spectral_quality_factor']):.12e}"),
        ("Circular polarization", f"{float(stokes.get('P_circ', np.nan)):.12e}"),
        ("Linear polarization", f"{float(stokes.get('P_lin', np.nan)):.12e}"),
        ("Radiation H3/H1", f"{float(r['harmonic_ratios']['H3_over_H1']):.12e}"),
        ("Radiation H5/H1", f"{float(r['harmonic_ratios']['H5_over_H1']):.12e}"),
        ("Gamma: initial / average", f"{float(r['gamma0']):.12e} / {float(r['gamma_avg']):.12e}"),
        ("Observer angle", f"{float(r.get('observer_theta_rad', 0.0))*1e3:.12e} mrad"),
    ]
    for i in range(0, len(summary_metrics), 2):
        cols = st.columns(2, gap="large")
        for j, (label, value) in enumerate(summary_metrics[i:i+2]):
            cols[j].metric(label, value)

    st.markdown("### Exact numerical summary")
    st.dataframe(
        _summary_table(r, stokes),
        width="stretch",
        hide_index=True,
        height=535,
    )

    # ---------------------------------------------------------------
    # 1. Magnetic field
    # ---------------------------------------------------------------
    st.markdown("## 1 · Magnetic field and field-integral diagnostics")
    field = tables.get("field")
    if isinstance(field, pd.DataFrame):
        zz = field["z_m"].to_numpy()
        bx = field["Bx_T"].to_numpy()
        by = field["By_T"].to_numpy()
        bz = field["Bz_T"].to_numpy()
        bp = field["Bperp_T"].to_numpy()

        for col, label in [
            ("Bx_T", "Bx"),
            ("By_T", "By"),
            ("Bz_T", "Bz"),
            ("Bperp_T", "|B⊥|"),
        ]:
            fig = go.Figure([_trace(field["z_m"], field[col], label)])
            _plot(st, fig, f"{label} along the magnetic axis", "z (m)", "B (T)")

        fig = go.Figure([
            _trace(field["z_m"], field["Bx_T"], "Bx"),
            _trace(field["z_m"], field["By_T"], "By"),
            _trace(field["z_m"], field["Bz_T"], "Bz"),
            _trace(field["z_m"], field["Bperp_T"], "|B⊥|"),
        ])
        _plot(st, fig, "All magnetic-field components", "z (m)", "B (T)", height=720)

        fig = go.Figure([
            go.Scatter(
                x=bx, y=by, mode="lines", name="Bx–By",
                hovertemplate="Bx=%{x:.10e} T<br>By=%{y:.10e} T<extra></extra>",
            )
        ])
        _plot(st, fig, "Transverse magnetic-field locus", "Bx (T)", "By (T)", height=700, hovermode="closest")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        phase = np.unwrap(np.arctan2(by, bx))
        fig = go.Figure([_trace(zz, phase, "atan2(By,Bx)")])
        _plot(st, fig, "Transverse-field phase evolution", "z (m)", "Unwrapped field phase (rad)")

        i1x = _cumtrap(bx, zz)
        i1y = _cumtrap(by, zz)
        i2x = _cumtrap(i1x, zz)
        i2y = _cumtrap(i1y, zz)

        fig = go.Figure([_trace(zz, i1x, "∫Bx dz"), _trace(zz, i1y, "∫By dz")])
        _plot(st, fig, "First magnetic-field integrals", "z (m)", "First field integral (T·m)")

        fig = go.Figure([_trace(zz, i2x, "∫∫Bx dz²"), _trace(zz, i2y, "∫∫By dz²")])
        _plot(st, fig, "Second magnetic-field integrals", "z (m)", "Second field integral (T·m²)")

        spatial_freq = np.fft.rfftfreq(len(zz), d=max(float(np.mean(np.diff(zz))), 1e-30))
        bx_fft = np.abs(np.fft.rfft(bx - np.mean(bx)))
        by_fft = np.abs(np.fft.rfft(by - np.mean(by)))
        sf_mask = spatial_freq > 0
        fig = go.Figure([
            _trace(spatial_freq[sf_mask], bx_fft[sf_mask], "|FFT(Bx)|"),
            _trace(spatial_freq[sf_mask], by_fft[sf_mask], "|FFT(By)|"),
        ])
        _plot(st, fig, "Spatial magnetic-field spectrum", "Spatial frequency (cycles/m)", "Amplitude")

        _show_table(
            st, "Magnetic field numerical table", field, "magnetic_field",
            caption="Float columns are shown with 10 digits after the decimal in scientific notation."
        )
    else:
        st.info("A magnetic-field table is unavailable for this completed result/device object.")

    fq = tables.get("field_quality")
    if isinstance(fq, dict):
        st.markdown("### Complete field-quality values")
        st.dataframe(_dict_df(fq), width="stretch", hide_index=True, height=520)

    # ---------------------------------------------------------------
    # 2. Trajectory
    # ---------------------------------------------------------------
    st.markdown("## 2 · Electron trajectory, orbit and phase-space diagnostics")

    fig = go.Figure([
        go.Scatter3d(
            x=pos[:, 0] * 1e3,
            y=pos[:, 1] * 1e3,
            z=pos[:, 2],
            mode="lines",
            name="electron",
            hovertemplate="x=%{x:.10e} mm<br>y=%{y:.10e} mm<br>z=%{z:.10e} m<extra></extra>",
        )
    ])
    fig.update_layout(
        title={"text": "3-D electron trajectory", "x": 0.01},
        height=PLOT_HEIGHT_3D,
        margin=dict(l=35, r=35, t=90, b=40),
        scene={
            "xaxis_title": "x (mm)",
            "yaxis_title": "y (mm)",
            "zaxis_title": "z (m)",
            "aspectmode": "data",
        },
        font=dict(size=15),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True, "responsive": True})

    for arr, name, unit_scale, unit in [
        (pos[:, 0], "x(z)", 1e3, "mm"),
        (pos[:, 1], "y(z)", 1e3, "mm"),
    ]:
        fig = go.Figure([_trace(z, arr * unit_scale, name)])
        _plot(st, fig, f"Electron transverse position {name}", "z (m)", f"Position ({unit})")

    fig = go.Figure([
        go.Scatter(
            x=pos[:, 0] * 1e3, y=pos[:, 1] * 1e3, mode="lines",
            name="orbit", hovertemplate="x=%{x:.10e} mm<br>y=%{y:.10e} mm<extra></extra>"
        )
    ])
    _plot(st, fig, "Transverse orbit locus", "x (mm)", "y (mm)", height=700, hovermode="closest")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    xp = vel[:, 0] / vz
    yp = vel[:, 1] / vz
    for arr, name in [(xp, "x′"), (yp, "y′")]:
        fig = go.Figure([_trace(z, arr * 1e3, name)])
        _plot(st, fig, f"Trajectory angle {name}(z)", "z (m)", f"{name} (mrad)")

    fig = go.Figure([
        go.Scatter(
            x=pos[:, 0] * 1e3, y=xp * 1e3, mode="lines",
            name="x–x′", hovertemplate="x=%{x:.10e} mm<br>x′=%{y:.10e} mrad<extra></extra>"
        )
    ])
    _plot(st, fig, "Horizontal phase-space projection", "x (mm)", "x′ (mrad)", height=700, hovermode="closest")

    fig = go.Figure([
        go.Scatter(
            x=pos[:, 1] * 1e3, y=yp * 1e3, mode="lines",
            name="y–y′", hovertemplate="y=%{x:.10e} mm<br>y′=%{y:.10e} mrad<extra></extra>"
        )
    ])
    _plot(st, fig, "Vertical phase-space projection", "y (mm)", "y′ (mrad)", height=700, hovermode="closest")

    for j, nm in enumerate(["βx", "βy", "βz"]):
        fig = go.Figure([_trace(z, beta[:, j], nm)])
        _plot(st, fig, f"{nm} along the trajectory", "z (m)", nm)

    fig = go.Figure([_trace(z, gam, "γ")])
    _plot(st, fig, "Lorentz factor along the device", "z (m)", "γ")

    fig = go.Figure([_trace(z, P, "P_Larmor")])
    _plot(st, fig, "Instantaneous Larmor radiated power", "z (m)", "Power (W)")

    cumulative_energy = _cumtrap(P, ts)
    fig = go.Figure([_trace(z, cumulative_energy, "∫P dt")])
    _plot(st, fig, "Cumulative radiated energy", "z (m)", "Energy (J)")

    phase_obj = r.get("trajectory_phase", {})
    if isinstance(phase_obj, dict):
        st.markdown("### Trajectory / phase diagnostics — exact values")
        st.dataframe(_dict_df(phase_obj), width="stretch", hide_index=True, height=520)

    _show_table(st, "Full trajectory numerical table", tables["trajectory"], "trajectory", height=680)

    # ---------------------------------------------------------------
    # 3. Observer
    # ---------------------------------------------------------------
    st.markdown("## 3 · Time-domain radiation at the observer")
    tfs = (tobs - tobs[0]) * 1e15
    emag = np.linalg.norm(E, axis=1)

    for arr, name in [(E1, "E1"), (E2, "E2"), (emag, "|E|")]:
        fig = go.Figure([_trace(tfs, arr, name)])
        _plot(st, fig, f"Observer waveform: {name}", "Observer time offset (fs)", "Electric field (V/m)")

    fig = go.Figure([
        _trace(tfs, E[:, 0], "Ex"),
        _trace(tfs, E[:, 1], "Ey"),
        _trace(tfs, E[:, 2], "Ez"),
    ])
    _plot(st, fig, "Cartesian observer electric-field components", "Observer time offset (fs)", "Electric field (V/m)", height=720)

    emag2 = np.sum(E * E, axis=1)
    inst_flux = float(v11.eps_0) * float(v11.c0) * emag2
    fluence = _cumtrap(inst_flux, tobs)

    fig = go.Figure([_trace(tfs, inst_flux, "ε₀c|E|²")])
    _plot(st, fig, "Instantaneous radiative energy flux", "Observer time offset (fs)", "Flux (W/m²)")

    fig = go.Figure([_trace(tfs, fluence, "Cumulative fluence")])
    _plot(st, fig, "Cumulative radiative fluence", "Observer time offset (fs)", "Fluence (J/m²)")

    fig = go.Figure([
        go.Scatter(
            x=E1, y=E2, mode="lines", name="polarization locus",
            hovertemplate="E1=%{x:.10e} V/m<br>E2=%{y:.10e} V/m<extra></extra>"
        )
    ])
    _plot(st, fig, "Time-domain polarization locus", "E1 (V/m)", "E2 (V/m)", height=700, hovermode="closest")

    if len(tp) == len(tobs) and np.any(np.isfinite(tp)):
        fig = go.Figure([_trace(tfs, (tp - tp[0]) * 1e15, "t_ret")])
        _plot(st, fig, "Retarded source time mapped to observer time", "Observer time offset (fs)", "Retarded source-time offset (fs)")

        delay = tobs - tp
        fig = go.Figure([_trace(tfs, delay * 1e9, "t_obs - t_ret")])
        _plot(st, fig, "Source-to-observer retardation delay", "Observer time offset (fs)", "Retardation delay (ns)")

    _show_table(st, "Observer-field numerical table", tables["observer"], "observer_field", height=680)

    # ---------------------------------------------------------------
    # 4. Spectrum
    # ---------------------------------------------------------------
    st.markdown("## 4 · Spectrum, photon energy and harmonics")
    power = np.maximum(fft, 0.0) ** 2

    positive = freq > 0
    fig = go.Figure([_trace(freq[positive], power[positive], "|FFT|²")])
    _plot(st, fig, "Radiation spectrum — linear scale", "Frequency (Hz)", "Spectral power proxy", height=720)

    logmask = (freq > 0) & (power > 0)
    if np.any(logmask):
        fig = go.Figure([_trace(freq[logmask], power[logmask], "|FFT|²")])
        _plot(st, fig, "Radiation spectrum — log/log view", "Frequency (Hz)", "Spectral power proxy", height=720)
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    band = (freq > 0.5 * float(r["f_expected"])) & (freq < 1.5 * float(r["f_expected"]))
    if np.any(band):
        pnorm = power[band] / max(float(np.max(power[band])), 1e-300)
        fig = go.Figure([_trace(freq[band] / float(r["f_expected"]), pnorm, "normalized spectrum")])
        _plot(st, fig, "Fundamental spectral window", "f / f_theory", "Normalized spectral power")
        fig.add_vline(
            x=float(r["f0"]) / float(r["f_expected"]),
            line_dash="dash",
            annotation_text="simulated f₀",
        )

    photon_e = float(v11.h_planck) * freq / float(v11.qe)
    emask = (photon_e > 0) & (power > 0)
    if np.any(emask):
        fig = go.Figure([_trace(photon_e[emask], power[emask], "spectral power")])
        _plot(st, fig, "Spectrum on photon-energy axis", "Photon energy (eV)", "Spectral power proxy", height=720)
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    hrs = r.get("harmonic_ratios", {})
    hdf = pd.DataFrame({
        "harmonic": ["H1", "H3", "H5"],
        "relative_amplitude": [
            1.0,
            float(hrs.get("H3_over_H1", np.nan)),
            float(hrs.get("H5_over_H1", np.nan)),
        ],
    })
    fig = go.Figure([
        go.Bar(
            x=hdf["harmonic"],
            y=hdf["relative_amplitude"],
            text=[_fmt_value(v, scientific=True) for v in hdf["relative_amplitude"]],
            textposition="outside",
            hovertemplate="%{x}<br>relative amplitude=%{y:.10e}<extra></extra>",
        )
    ])
    _plot(st, fig, "Radiation harmonic ratios", "Harmonic", "Relative amplitude", height=600, hovermode="closest")

    _show_table(st, "Spectrum numerical table", tables["spectrum"], "spectrum", height=680)

    # ---------------------------------------------------------------
    # 5. Polarization
    # ---------------------------------------------------------------
    st.markdown("## 5 · Polarization and Stokes diagnostics")
    I = float(stokes.get("I", 0.0))
    denom = max(abs(I), 1e-300)
    qn = float(stokes.get("Q", 0.0)) / denom
    un = float(stokes.get("U", 0.0)) / denom
    vn = float(stokes.get("V", 0.0)) / denom
    sdf = pd.DataFrame({
        "parameter": ["I", "Q/I", "U/I", "V/I", "P_lin", "P_circ"],
        "value": [
            I, qn, un, vn,
            float(stokes.get("P_lin", 0.0)),
            float(stokes.get("P_circ", 0.0)),
        ],
    })

    fig = go.Figure([
        go.Bar(
            x=sdf["parameter"],
            y=sdf["value"],
            text=[_fmt_value(v, scientific=True) for v in sdf["value"]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.10e}<extra></extra>",
        )
    ])
    _plot(st, fig, "Stokes / polarization summary", "Parameter", "Value", height=650, hovermode="closest")

    theta = np.linspace(0, 2 * np.pi, 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines", name="P_lin = 1 boundary"))
    fig.add_trace(go.Scatter(
        x=[qn], y=[un], mode="markers+text", name="Stokes point",
        text=[f"V/I={vn:.8e}"], textposition="top center",
        marker={"size": 14},
        hovertemplate="Q/I=%{x:.10e}<br>U/I=%{y:.10e}<extra></extra>",
    ))
    _plot(st, fig, "Normalized Q–U polarization plane", "Q/I", "U/I", height=700, hovermode="closest")
    fig.update_xaxes(range=[-1.1, 1.1])
    fig.update_yaxes(range=[-1.1, 1.1], scaleanchor="x", scaleratio=1)

    st.dataframe(
        sdf,
        width="stretch",
        hide_index=True,
        height=350,
        column_config=_numeric_column_config(st, sdf),
    )

    # ---------------------------------------------------------------
    # 6. Energy / photon / theory
    # ---------------------------------------------------------------
    st.markdown("## 6 · Energy accounting, photon yield and theory checks")
    dict_sections = [
        ("Energy accounting", "energy_accounting"),
        ("Photon-yield estimates", "photon_yield"),
        ("Spectral photon-yield estimate", "spectral_photon_yield"),
        ("Theory residuals", "theory_residuals"),
        ("Pulse diagnostics", "pulse"),
        ("Orbit summary", "traj"),
        ("K components", "K_components"),
    ]
    for title, key in dict_sections:
        obj = r.get(key, {})
        if isinstance(obj, dict):
            st.markdown(f"### {title}")
            ddf = _dict_df(obj)
            st.dataframe(ddf, width="stretch", hide_index=True, height=min(560, 130 + 45 * max(1, len(ddf))))
            numeric_rows = []
            for k, v in obj.items():
                if isinstance(v, (int, float, np.number)) and np.isfinite(float(v)):
                    numeric_rows.append((str(k), float(v)))
            if numeric_rows and len(numeric_rows) <= 14:
                fig = go.Figure([
                    go.Bar(
                        x=[x[0] for x in numeric_rows],
                        y=[x[1] for x in numeric_rows],
                        text=[_fmt_value(x[1], scientific=True) for x in numeric_rows],
                        textposition="outside",
                        hovertemplate="%{x}<br>%{y:.10e}<extra></extra>",
                    )
                ])
                _plot(st, fig, f"{title} — numerical overview", "Quantity", "Value", height=620, hovermode="closest")

    # ---------------------------------------------------------------
    # 7. Quantum
    # ---------------------------------------------------------------
    st.markdown("## 7 · Quantum monitor")
    qdf = tables.get("quantum")
    if isinstance(qdf, pd.DataFrame):
        fig = go.Figure([_trace(qdf["z_m"], qdf["chi_e"], "χe")])
        _plot(st, fig, "Strong-field quantum parameter χe", "z (m)", "χe")

        fig = go.Figure([_trace(qdf["z_m"], qdf["gaunt_factor"], "Gaunt factor")])
        _plot(st, fig, "Quantum Gaunt factor", "z (m)", "Gaunt factor")

        fig = go.Figure([
            go.Scatter(
                x=qdf["chi_e"], y=qdf["gaunt_factor"], mode="lines",
                name="trajectory", hovertemplate="χe=%{x:.10e}<br>g=%{y:.10e}<extra></extra>"
            )
        ])
        _plot(st, fig, "Gaunt factor versus χe", "χe", "Gaunt factor", height=700, hovermode="closest")

        _show_table(st, "Quantum numerical table", qdf, "quantum", height=620)
    else:
        st.info("Quantum trajectory arrays are unavailable for this result.")

    # ---------------------------------------------------------------
    # 8. Complete scalar inventory
    # ---------------------------------------------------------------
    st.markdown("## 8 · Complete scalar result inventory")
    scalar_df = tables["scalars"].copy()
    if "value" in scalar_df:
        scalar_df["value"] = scalar_df["value"].map(
            lambda x: _fmt_value(x, scientific=True)
            if isinstance(x, (int, float, np.number)) else str(x)
        )
    st.dataframe(scalar_df, width="stretch", height=760, hide_index=True)
    st.download_button(
        "Download scalar-result inventory CSV",
        scalar_df.to_csv(index=False).encode("utf-8"),
        file_name="scalar_result_inventory.csv",
        mime="text/csv",
        key="download_fullresults_scalars",
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # 9. Export
    # ---------------------------------------------------------------
    st.markdown("## 9 · Complete result export")
    st.caption(
        "The ZIP contains trajectory, observer field, spectrum, quantum arrays, "
        "magnetic field, scalar inventory and nested diagnostic JSON files."
    )
    st.download_button(
        "Download complete result data bundle (.zip)",
        result_bundle_zip(r, v11, dev, field_quality_fn),
        file_name="v11_complete_numerical_results.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_fullresults_bundle",
    )
