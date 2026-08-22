from __future__ import annotations
import numpy as np
from radia_support import load_radia
from devices.factory import build_device
from solver.pipeline import solve_model, sample_3d
from analysis.geometry_bounds import geometry_field_range
from calibration.target_b0 import calibrate_br

DEVICE_MAP = {
    "planar": "Planar",
    "helical": "Helical",
    "left_helical": "Helical",
    "elliptical": "Elliptical",
    "variable_polarization": "APPLE-II",
    "apple2": "APPLE-II",
    "wiggler": "Wiggler",
}

def _switch(enabled, value):
    return value if enabled else 0.0

def default_parameters(
    device_name, lambda_u_m, n_periods, *,
    gap_m=0.012, block_width_m=None, block_height_m=0.015,
    br_t=1.20, error_switches=None, error_config=None,
    error_seed=20260820, material_mode="Fixed remanence",
    mu_parallel=1.05, mu_perpendicular=1.05, segmentation=(1,1,1),
    ellipticity=0.50, apple_phase_deg=90.0,
    apple_shift_mode="Antiparallel",
):
    if device_name not in DEVICE_MAP:
        raise ValueError(
            f"Unsupported RADIA-generated device {device_name!r}. "
            f"Available: {', '.join(DEVICE_MAP)}"
        )
    switches = {
        "field_amplitude": True,
        "longitudinal_position": True,
        "transverse_position": True,
        "magnetization_angle": True,
        "gap_asymmetry": True,
        "bank_strength_imbalance": True,
    }
    if error_switches is not None:
        switches.update({k:bool(v) for k,v in error_switches.items() if k in switches})

    cfg = {
        "field_amplitude":{"rms_fraction":0.002},
        "longitudinal_position":{"rms_m":20e-6},
        "transverse_position":{"rms_m":10e-6},
        "magnetization_angle":{"rms_rad":0.5e-3},
        "gap_asymmetry":{"rms_m":10e-6},
        "bank_strength_imbalance":{"rms_fraction":0.001},
    }
    if error_config:
        for k,v in error_config.items():
            if k in cfg and isinstance(v,dict):
                cfg[k].update(v)

    kind=DEVICE_MAP[device_name]
    if block_width_m is None:
        block_width_m = 0.010 if kind in ("Helical","Elliptical") else 0.040
    return {
        "device":kind,
        "period_mm":float(lambda_u_m)*1e3,
        "periods":int(n_periods),
        "gap_mm":float(gap_m)*1e3,
        "blocks_per_period":8 if kind in ("Helical","Elliptical") else 4,
        "block_width_mm":float(block_width_m)*1e3,
        "block_height_mm":float(block_height_m)*1e3,
        "longitudinal_fill":0.90,
        "br_t":float(br_t),
        "material_mode":material_mode,
        "mu_parallel":float(mu_parallel),
        "mu_perpendicular":float(mu_perpendicular),
        "segmentation":tuple(int(v) for v in segmentation),
        "ellipticity":float(ellipticity),
        "apple_phase_deg":float(apple_phase_deg),
        "apple_shift_mode":str(apple_shift_mode),
        "handedness":-1 if device_name=="left_helical" else 1,
        "errors_enabled":any(switches.values()),
        "field_error_pct":100.0*_switch(switches["field_amplitude"],float(cfg["field_amplitude"]["rms_fraction"])),
        "longitudinal_error_mm":1e3*_switch(switches["longitudinal_position"],float(cfg["longitudinal_position"]["rms_m"])),
        "transverse_error_mm":1e3*_switch(switches["transverse_position"],float(cfg["transverse_position"]["rms_m"])),
        "angle_error_deg":float(np.degrees(_switch(switches["magnetization_angle"],float(cfg["magnetization_angle"]["rms_rad"])))),
        "gap_asymmetry_mm":1e3*_switch(switches["gap_asymmetry"],float(cfg["gap_asymmetry"]["rms_m"])),
        "bank_imbalance_pct":100.0*_switch(switches["bank_strength_imbalance"],float(cfg["bank_strength_imbalance"]["rms_fraction"])),
        "error_seed":int(error_seed),
    }

def build_radia_map(
    device_name, lambda_u_m, n_periods, target_B0_T, *,
    gap_m=0.012, block_width_m=None, block_height_m=0.015,
    x_half_m=0.003, y_half_m=0.003, nx=7, ny=7,
    samples_per_period=24, field_margin_periods=1.0,
    error_switches=None, error_config=None, error_seed=20260820,
    material_mode="Fixed remanence", mu_parallel=1.05,
    mu_perpendicular=1.05, segmentation=(1,1,1),
    ellipticity=0.50, apple_phase_deg=90.0,
    apple_shift_mode="Antiparallel", rad=None,
):
    if rad is None:
        rad=load_radia()

    p=default_parameters(
        device_name,lambda_u_m,n_periods,
        gap_m=gap_m,block_width_m=block_width_m,block_height_m=block_height_m,
        error_switches=error_switches,error_config=error_config,error_seed=error_seed,
        material_mode=material_mode,mu_parallel=mu_parallel,
        mu_perpendicular=mu_perpendicular,segmentation=segmentation,
        ellipticity=ellipticity,apple_phase_deg=apple_phase_deg,
        apple_shift_mode=apple_shift_mode,
    )
    relax=material_mode=="Linear NdFeB + relaxation"

    # Calibrate on ideal geometry only.
    p_cal=dict(p); p_cal["errors_enabled"]=False
    calibrated_br,history=calibrate_br(
        rad,p["device"],p_cal,float(target_B0_T),
        mode="Central-period peak B⊥",
        relax=relax,precision=1e-4,max_iter=1000,
        samples=max(161,int(samples_per_period)*8+1),
    )
    p["br_t"]=float(calibrated_br)

    if hasattr(rad,"UtiDelAll"):
        rad.UtiDelAll()
    model=build_device(rad,p["device"],p)
    rlx=solve_model(rad,model,relax=relax,precision=1e-4,max_iter=1000,method=4)

    zlo,zhi=geometry_field_range(model["blocks"],p["period_mm"],float(field_margin_periods))
    x=np.linspace(-float(x_half_m)*1e3,float(x_half_m)*1e3,int(nx))
    y=np.linspace(-float(y_half_m)*1e3,float(y_half_m)*1e3,int(ny))
    span_periods=max(1.0,(zhi-zlo)/float(p["period_mm"]))
    nz=max(81,int(np.ceil(span_periods*int(samples_per_period)))+1)
    z=np.linspace(zlo,zhi,nz)

    Bzyx=sample_3d(rad,model["obj"],x,y,z)
    Bxyz=np.transpose(Bzyx,(2,1,0,3))

    blocks=model["blocks"]
    geom_lo=min(float(b["center"][2])-0.5*float(b["size"][2]) for b in blocks)
    geom_hi=max(float(b["center"][2])+0.5*float(b["size"][2]) for b in blocks)

    return {
        "x_m":x*1e-3,"y_m":y*1e-3,"z_m":z*1e-3,
        "Bx_T":Bxyz[:,:,:,0],"By_T":Bxyz[:,:,:,1],"Bz_T":Bxyz[:,:,:,2],
        "blocks":blocks,
        "parameters":p,
        "metadata":{
            "backend":"RADIA Magnet Studio strict backend",
            "device_kind":p["device"],
            "device_name":device_name,
            "device_length_m":float(lambda_u_m)*int(n_periods),
            "geometry_z_edges_m":[geom_lo*1e-3,geom_hi*1e-3],
            "tracking_z_start_m":float(z[0])*1e-3,
            "tracking_z_end_m":float(z[-1])*1e-3,
            "field_margin_periods":float(field_margin_periods),
            "target_B0_T":float(target_B0_T),
            "calibrated_Br_T":float(calibrated_br),
            "calibration_history":history,
            "calibration_relative_error":float(history[-1]["relative_error"]),
            "calibration_verified":bool(history[-1]["relative_error"] <= 5e-3),
            "remanence_plausibility_warning":(
                None if float(calibrated_br) <= 1.6 else
                f"Calibrated Br={float(calibrated_br):.4g} T exceeds a typical NdFeB remanence range; review geometry/material assumptions."
            ),
            "same_Br_for_error_model":True,
            "relaxation_result":rlx,
            "grid_shape_xyz":[len(x),len(y),len(z)],
            "error_switches":dict(error_switches or {}),
        },
    }
