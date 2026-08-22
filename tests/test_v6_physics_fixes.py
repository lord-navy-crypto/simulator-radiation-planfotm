import sys, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import undulator_v11_radia_integrated_v9 as v11


def make_synthetic(device_name, Bx, By, lam=0.05, n=20):
    x=np.array([-1e-3,0,1e-3]); y=x.copy()
    z=np.linspace(-0.05,n*lam+0.05,n*48+97)
    shape=(3,3,len(z)); bx=np.zeros(shape); by=np.zeros(shape); bz=np.zeros(shape)
    phase=2*np.pi*(z-0.5*n*lam)/lam
    env=np.where((z>=0)&(z<=n*lam),1.0,0.0)
    bx[:]=Bx*np.sin(phase)[None,None,:]*env[None,None,:]
    by[:]=By*np.cos(phase)[None,None,:]*env[None,None,:]
    return v11.FieldMapInsertionDevice(
        x,y,z,bx,by,bz,lambda_u=lam,device_name=device_name,
        metadata={"device_length_m":n*lam,"geometry_z_edges_m":[0.0,n*lam],"tracking_z_start_m":float(z[0]),"tracking_z_end_m":float(z[-1])}
    )


def run():
    # Generalized resonance: planar must use 1 + K^2/2, not helical 1 + K^2.
    planar=make_synthetic("planar",0.0,0.20)
    gamma=100.0
    kc=planar.K_components(v11.me)
    K=kc["Ky"] if kc["Ky"]>kc["Kx"] else kc["Kx"]
    lam_expected=planar.lambda_u*(1+0.5*K*K)/(2*gamma*gamma)
    assert abs(v11.fund_lambda_device(gamma,planar)-lam_expected)/lam_expected < 5e-4

    # Generalized power K_eff: planar K/sqrt2; circular helical K.
    kp=v11.device_power_K_eff(planar)
    assert abs(kp-K/math.sqrt(2))/max(K,1e-30) < 5e-3
    hel=make_synthetic("helical",0.20,0.20)
    kh=hel.K_components(v11.me)
    expected_h=0.5*(kh["Kx"]+kh["Ky"])
    assert abs(v11.device_power_K_eff(hel)-expected_h)/expected_h < 5e-3

    # FieldMap K components are based on central fundamental, not global fringe peak.
    assert hasattr(planar,"Btrans_peak_global") and hasattr(planar,"By_fundamental")
    assert planar.metadata["characterization_z_range_m"][0] > planar.z_grid[0]

    # Wiggler generated default target must actually be passed as the K=5 B0.
    target=v11.B0_from_K(5.0,0.05)
    assert 0.9 < target < 1.2
    seen={}
    original=v11.generate_radia_field_device
    try:
        def fake_generate(**kwargs):
            seen.update(kwargs)
            return kwargs
        v11.generate_radia_field_device=fake_generate
        v11.make_default_undulator(
            preset="wiggler",field_model="radia_generated",n_periods=5,radia_options={}
        )
    finally:
        v11.generate_radia_field_device=original
    assert abs(seen["target_B0_T"]-target)/target < 1e-12

    # Exact terminal event: final z must equal map z_end to numerical precision.
    span=v11.simulation_span_for_device(gamma,planar,n_periods=20)
    state,_=v11.make_initial_state_device(gamma,planar)
    npts=3000
    sol=v11.solve_ivp(
        v11.rhs_lorentz,span,state,args=(planar,v11.me,-v11.qe),
        t_eval=np.linspace(*span,npts),rtol=1e-9,atol=1e-11,
        events=(v11.aperture_event,v11.field_map_end_event)
    )
    ts,Y=v11.solution_arrays_with_terminal_sample(sol)
    assert len(sol.t_events)>1 and len(sol.t_events[1])==1
    assert abs(Y[2,-1]-planar.metadata["tracking_z_end_m"]) < 1e-10

    print("V6 SIX-PHYSICS-FIX REGRESSION TEST PASSED")

if __name__=='__main__': run()
