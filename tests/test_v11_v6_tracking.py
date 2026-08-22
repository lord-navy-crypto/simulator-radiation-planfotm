import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import undulator_v11_radia_integrated_v9 as v11

def run():
    x=np.array([-1e-3,0.0,1e-3])
    y=np.array([-1e-3,0.0,1e-3])
    z=np.linspace(-0.05,0.15,21)
    shape=(len(x),len(y),len(z))
    bx=np.zeros(shape)
    by=np.zeros(shape)
    bz=np.zeros(shape)
    by[:]=0.10*np.sin(2*np.pi*z[None,None,:]/0.05)

    dev=v11.FieldMapInsertionDevice(
        x,y,z,bx,by,bz,
        lambda_u=0.05,
        metadata={
            "device_length_m":0.10,
            "tracking_z_start_m":-0.05,
            "tracking_z_end_m":0.15,
        },
    )
    state,_=v11.make_initial_state_device(100.0,dev)
    assert abs(state[2]+0.05)<1e-12

    span=v11.simulation_span_for_device(100.0,dev,n_periods=2)
    beta=v11.ideal_beta_z_device(100.0,dev)
    nominal=0.20/(beta*v11.c0)
    assert (span[1]-span[0]) >= nominal
    assert (span[1]-span[0]) <= 1.25*nominal

    print("V11 V6 FULL REAL-MAP TRACKING RANGE TEST PASSED")

if __name__=="__main__":
    run()
