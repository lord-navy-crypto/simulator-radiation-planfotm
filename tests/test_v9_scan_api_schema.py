import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import undulator_v11_radia_integrated_v9 as v11

def fake_result():
    return {
        'f0':1e14,'photon_energy':{'eV':1.0},'P_larmor':2.5,
        'relative_linewidth':0.02,'spectral_fwhm_hz':2e12,
        'spectral_quality_factor':50.0,
        'harmonic_ratios':{'H3_over_H1':0.1,'H5_over_H1':0.01},
        'Stokes':{'P_circ':0.2,'P_lin':0.8},
        'K_components':{'Kx':0.5,'Ky':0.0},
        'photon_yield':{'equivalent_photons':3.0},
    }

def run():
    old_run=v11.run_sim
    old_span=v11.simulation_span_for_device
    old_beta=v11.ideal_beta_z_device
    old_field=v11.FIELD_MODEL
    try:
        v11.run_sim=lambda *a,**k: fake_result()
        v11.simulation_span_for_device=lambda *a,**k:(0.0,1.0)
        v11.ideal_beta_z_device=lambda *a,**k:0.99
        v11.FIELD_MODEL='analytic'
        a=v11.k_scan(100.0,K_values=(0.2,),n_periods=2,realistic=False,n_base=256)
        b=v11.period_number_scan(100.0,N_values=(2,),realistic=False)
        c=v11.compare_device_presets(100.0,preset_names=('planar',),n_periods=2,realistic=False)
        assert a[0]['power_W']==2.5
        assert b[0]['power_W']==2.5
        assert c[0]['power_W']==2.5
    finally:
        v11.run_sim=old_run
        v11.simulation_span_for_device=old_span
        v11.ideal_beta_z_device=old_beta
        v11.FIELD_MODEL=old_field
    print('V9 SCAN API SCHEMA TEST PASSED')
if __name__=='__main__':run()
