import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))

from fake_radia import FakeRadia
from v11_radia_backend_v8 import build_radia_map

SWITCHES={
    "field_amplitude":True,
    "longitudinal_position":True,
    "transverse_position":True,
    "magnetization_angle":True,
    "gap_asymmetry":True,
    "bank_strength_imbalance":True,
}

def run():
    for name in [
        "planar","helical","left_helical","elliptical",
        "variable_polarization","apple2","wiggler"
    ]:
        rad=FakeRadia()
        m=build_radia_map(
            name,0.05,2,0.15,
            nx=3,ny=3,samples_per_period=6,
            x_half_m=0.001,y_half_m=0.001,
            error_switches=SWITCHES,rad=rad
        )
        assert m["Bx_T"].shape==m["By_T"].shape==m["Bz_T"].shape
        assert m["Bx_T"].shape[:2]==(3,3)
        assert m["Bx_T"].shape[2]==len(m["z_m"])
        assert m["metadata"]["tracking_z_start_m"] < m["metadata"]["tracking_z_end_m"]
        assert m["metadata"]["same_Br_for_error_model"] is True
        assert m["metadata"]["calibrated_Br_T"] > 0
        assert m["blocks"]

    # Handedness must change the generated magnetization sequence.
    off={k:False for k in SWITCHES}
    h=build_radia_map("helical",0.05,1,0.15,nx=3,ny=3,samples_per_period=6,rad=FakeRadia(),error_switches=off)
    l=build_radia_map("left_helical",0.05,1,0.15,nx=3,ny=3,samples_per_period=6,rad=FakeRadia(),error_switches=off)
    axes_h=np.asarray([b["axis"] for b in h["blocks"]],float)
    axes_l=np.asarray([b["axis"] for b in l["blocks"]],float)
    assert not np.allclose(axes_h,axes_l)

    print("V6 RADIA BRIDGE MOCK TEST PASSED")

if __name__=="__main__":
    run()
