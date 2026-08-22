import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import undulator_v11_radia_integrated_v9 as v11

def run():
    # Fluence has physical units and is time-integrated (not peak E^2).
    t=np.linspace(0.0,2.0,2001)
    E=np.zeros((len(t),3)); E[:,0]=3.0
    got=v11.radiative_fluence_J_m2(t,E)
    expected=v11.eps_0*v11.c0*(3.0**2)*2.0
    assert abs(got-expected)/expected < 1e-12,(got,expected)

    und=v11.make_default_undulator(realistic=False,preset='planar',field_model='analytic',n_periods=12)
    span=v11.simulation_span_for_device(100.0,und,n_periods=12)
    res=v11.run_sim(und,None,span,np.array([0.0,0.0,100.0]),n_base=768,gamma0_input=100.0)

    sc=v11.angle_scan(res,np.array([-2e-3,0.0,2e-3]),n_obs=3500)
    assert sc.shape==(3,5)
    assert np.all(np.isfinite(sc[:,2]))
    assert np.all(sc[:,2]>0.0)
    div=v11.divergence_from_angle_scan(sc)
    assert np.isfinite(div['rms_divergence_rad'])

    amap=v11.angular_map_2d(res,gamma_for_grid=100.0,grid_points=5,extent_gamma_theta=1.0,observer_distance=100.0,n_obs=2200)
    assert amap['intensity_quantity']=='radiative_fluence_J_m2'
    assert amap['fluence_J_m2'].shape==(5,5)
    assert amap['valid_mask'].shape==(5,5)
    assert amap['failure_count']==len(amap['failures'])
    assert np.all(np.isnan(amap['fluence_J_m2'][~amap['valid_mask']]))
    assert np.all(amap['fluence_J_m2'][amap['valid_mask']]>=0.0)

    # Legacy public scan helpers must use current P_larmor result schema.
    src=Path(ROOT/'undulator_v11_radia_integrated_v9.py').read_text()
    assert 'res["Pl"]' not in src
    assert '"power_W": float(res["P_larmor"])' in src

    print('V9 FINAL ANALYSIS TEST PASSED')
if __name__=='__main__':run()
