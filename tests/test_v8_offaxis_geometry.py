import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))

import undulator_v11_radia_integrated_v9 as v11
from fake_radia import FakeRadia
from devices.factory import build_device
from v11_radia_backend_v8 import build_radia_map

def aabb_overlap(a,b,tol=1e-10):
    ca=np.asarray(a["center"],float); sa=np.asarray(a["size"],float)
    cb=np.asarray(b["center"],float); sb=np.asarray(b["size"],float)
    alo=ca-sa/2; ahi=ca+sa/2
    blo=cb-sb/2; bhi=cb+sb/2
    overlap=np.minimum(ahi,bhi)-np.maximum(alo,blo)
    return bool(np.all(overlap > tol))

def interrow_collisions(blocks):
    hits=[]
    for i,a in enumerate(blocks):
        for j in range(i+1,len(blocks)):
            b=blocks[j]
            if a.get("row")==b.get("row"):
                continue
            if aabb_overlap(a,b):
                hits.append((i,j,a.get("row"),b.get("row")))
    return hits

BASE={
    "period_mm":50.0,"periods":1,"gap_mm":12.0,"blocks_per_period":8,
    "block_width_mm":10.0,"block_height_mm":15.0,"longitudinal_fill":0.90,
    "br_t":1.2,"material_mode":"Fixed remanence","mu_parallel":1.05,
    "mu_perpendicular":1.05,"segmentation":(1,1,1),"ellipticity":0.5,
    "apple_phase_deg":90.0,"apple_shift_mode":"Antiparallel",
    "handedness":1,"errors_enabled":False,
}

def run():
    # Geometry: no positive-volume inter-row collision at supported defaults.
    for kind in ("Helical","Elliptical"):
        p=dict(BASE)
        m=build_device(FakeRadia(),kind,p)
        hits=interrow_collisions(m["blocks"])
        assert not hits,(kind,hits[:5])

        # side blocks must have radial thickness in x and tangential width in y.
        right=next(b for b in m["blocks"] if b["row"]=="right")
        assert np.allclose(right["size"][:2],[15.0,10.0])

    p=dict(BASE); p["blocks_per_period"]=4; p["block_width_mm"]=40.0
    apple=build_device(FakeRadia(),"APPLE-II",p)
    hits=interrow_collisions(apple["blocks"])
    assert not hits,hits[:5]
    tr=next(b for b in apple["blocks"] if b["row"]=="TR")
    assert abs(tr["center"][0]-(12.0/2+40.0/2))<1e-12
    assert abs(tr["center"][1]-(12.0/2+15.0/2))<1e-12

    # Invalid cross-bank width is rejected, instead of silently overlapping.
    bad=dict(BASE); bad["block_width_mm"]=40.0
    try:
        build_device(FakeRadia(),"Helical",bad)
        raise AssertionError("overlapping helical geometry was not rejected")
    except ValueError as exc:
        assert "must be <= magnetic gap" in str(exc)

    # Bridge defaults for helical must now be non-overlapping.
    m=build_radia_map(
        "helical",0.05,1,0.15,
        nx=3,ny=3,samples_per_period=6,
        rad=FakeRadia(),
        error_switches={
            "field_amplitude":False,"longitudinal_position":False,
            "transverse_position":False,"magnetization_angle":False,
            "gap_asymmetry":False,"bank_strength_imbalance":False,
        }
    )
    assert not interrow_collisions(m["blocks"])

    # Off-axis theoretical formula and peak selection in full scalar simulation.
    und=v11.make_default_undulator(
        realistic=False,preset="planar",field_model="analytic",n_periods=20
    )
    theta=5e-3
    ro=np.array([100.0*np.tan(theta),0.0,100.0])
    span=v11.simulation_span_for_device(100.0,und,n_periods=20)
    rr=v11.run_sim_scalar(
        und,None,span,ro,n_base=1280,gamma0_input=100.0
    )
    theory=v11.fund_freq_device(rr["gamma_avg"],und,theta=theta)
    assert abs(rr["f_expected"]-theory)/theory < 1e-10
    assert abs(rr["f0"]-theory)/theory < 0.02, (rr["f0"],theory)

    # 1-D angular scan must follow red-shift rather than reusing on-axis timing.
    full=v11.run_sim(
        und,None,span,np.array([0.0,0.0,100.0]),
        n_base=1280,gamma0_input=100.0
    )
    scan=v11.angle_scan(full,np.array([0.0,2e-3,5e-3]),n_obs=5000)
    assert scan[1,1] < scan[0,1]
    assert scan[2,1] < scan[1,1]
    for row in scan:
        th=float(row[0])
        ro2=np.array([100.0*np.tan(th),0.0,100.0])
        fth=v11.expected_frequency_from_result(full,ro2)
        assert abs(row[1]-fth)/fth < 0.03,(th,row[1],fth)

    # Tracking resolution: hidden 96-points/period floor removed.
    source=Path(ROOT/"undulator_v11_radia_integrated_v9.py").read_text()
    run_chunk=source[source.index("def run_sim("):source.index("def angular_divergence(")]
    assert "pts_per_period=96" not in run_chunk
    assert "n_coarse = max(256, int(n_base))" in run_chunk

    print("V8 OFF-AXIS / GEOMETRY / RESOLUTION TEST PASSED")

if __name__=="__main__":
    run()
