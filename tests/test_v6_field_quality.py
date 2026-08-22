import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import undulator_v11_radia_integrated_v9 as v11
from v11_field_quality_v6 import field_quality_metrics


def run():
    lam=0.05; n=12
    x=np.array([-1e-3,0,1e-3]); y=x.copy(); z=np.linspace(-lam,n*lam+lam,1201)
    phase=2*np.pi*z/lam
    main=((z>=0)&(z<=n*lam)).astype(float)
    fringe=((z<0)|(z>n*lam)).astype(float)
    # Central device is pure fundamental. Third harmonic exists only in fringe region.
    by_line=0.2*np.cos(phase)*main + 0.08*np.cos(3*phase)*fringe
    shape=(3,3,len(z)); bx=np.zeros(shape); by=np.zeros(shape); bz=np.zeros(shape)
    by[:]=by_line[None,None,:]
    dev=v11.FieldMapInsertionDevice(
        x,y,z,bx,by,bz,lambda_u=lam,device_name='planar',
        metadata={'device_length_m':n*lam,'geometry_z_edges_m':[0.0,n*lam],
                  'tracking_z_start_m':float(z[0]),'tracking_z_end_m':float(z[-1])}
    )
    fq=field_quality_metrics(dev,n)
    assert fq['magnetic_H3_over_H1'] < 1e-2, fq['magnetic_H3_over_H1']
    assert abs(fq['By_fundamental_T']-0.2) < 2e-3
    assert fq['full_z_start_m'] < 0 and fq['full_z_end_m'] > n*lam
    assert np.isfinite(fq['first_integral_By_Tm']) and np.isfinite(fq['second_integral_By_Tm2'])
    print('V6 CENTRAL-HARMONICS / FULL-INTEGRALS TEST PASSED')

if __name__=='__main__': run()
