import sys, tempfile, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))

import undulator_v11_radia_integrated_v9 as v11
from fake_radia import FakeRadia
from calibration.target_b0 import calibrate_br

def run():
    # Causal retarded-time solution must be accurate and monotonic.
    beta=0.999
    ts=np.linspace(0.0,3e-9,200)
    r=np.zeros((len(ts),3))
    r[:,2]=beta*v11.c0*ts
    v=np.zeros_like(r)
    v[:,2]=beta*v11.c0
    rs=CubicSpline(ts,r,axis=0)
    vs=CubicSpline(ts,v,axis=0)
    ro=np.array([0.0,0.0,100.0])

    true_t=np.linspace(1e-12,2.5e-9,60)
    t_obs=100.0/v11.c0 + true_t*(1.0-beta)
    solved=np.array([
        v11.brent_solve(to,ro,ts[0],ts[-1],rs,vs)
        for to in t_obs
    ])
    assert np.all(np.diff(solved)>0)
    assert np.max(np.abs(solved-true_t)) < 5e-18

    # CSV branch: no NameError and arbitrary period is preserved.
    lam=0.03
    x=np.array([-1e-3,0,1e-3]); y=x.copy()
    z=np.linspace(-0.03,0.09,13)
    rows=[]
    for xx in x:
        for yy in y:
            for zz in z:
                rows.append({
                    "x_m":xx,"y_m":yy,"z_m":zz,
                    "Bx_T":0.0,
                    "By_T":0.2*np.sin(2*np.pi*zz/lam),
                    "Bz_T":0.0,
                })
    df=pd.DataFrame(rows)
    dev=v11._field_map_from_dataframe(
        df,lambda_u=lam,device_name="planar",
        shift_z_to_zero=False
    )
    assert abs(dev.lambda_u-lam)<1e-15

    fd,path=tempfile.mkstemp(suffix=".csv"); os.close(fd)
    try:
        df.to_csv(path,index=False)
        dev2=v11.make_default_undulator(
            preset="planar",field_model="radia_csv",
            n_periods=4,radia_csv_path=path,
            radia_csv_lambda_u=lam,
        )
        assert abs(dev2.lambda_u-lam)<1e-15
    finally:
        os.unlink(path)

    # High target: multiple bounded calibration steps must actually reach target.
    params={
        "period_mm":50.0,"periods":2,"gap_mm":12.0,
        "blocks_per_period":4,"block_width_mm":10.0,
        "block_height_mm":10.0,"longitudinal_fill":0.9,
        "br_t":1.2,"material_mode":"Fixed remanence",
        "mu_parallel":1.05,"mu_perpendicular":1.05,
        "segmentation":(1,1,1),"ellipticity":0.5,
        "apple_phase_deg":90.0,"apple_shift_mode":"Antiparallel",
        "errors_enabled":False,
    }
    target=v11.B0_from_K(5.0,0.05)
    br,hist=calibrate_br(
        FakeRadia(),"Planar",params,target,
        mode="Central-period peak B⊥",
        samples=201,relative_tolerance=5e-3
    )
    assert hist[-1]["relative_error"] <= 5e-3
    assert abs(hist[-1]["B0_T"]-target)/target <= 5e-3
    assert br > 1.2

    print("V7 RUNTIME CORRECTNESS TEST PASSED")

if __name__=="__main__":
    run()
