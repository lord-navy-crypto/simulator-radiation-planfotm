from __future__ import annotations
import numpy as np


def _cumtrap(y, x):
    y=np.asarray(y,float); x=np.asarray(x,float)
    out=np.zeros_like(y,dtype=float)
    if len(y)>1:
        out[1:]=np.cumsum(0.5*(y[1:]+y[:-1])*np.diff(x))
    return out


def _harmonic_amplitude(z, signal, ku, n):
    z=np.asarray(z,float); signal=np.asarray(signal,float)
    M=np.column_stack([np.sin(n*ku*z),np.cos(n*ku*z),np.ones_like(z)])
    coef,*_=np.linalg.lstsq(M,signal,rcond=None)
    return float(np.hypot(coef[0],coef[1]))


def field_quality_metrics(dev, nper):
    """Central periodic field quality + full fringe-field integrals.

    Central region:
      fundamental Bx/By, H3/H1, H5/H1, central Bperp peak.
    Full map/device:
      global Bperp/Bz peaks, I1 and I2.
    """
    if getattr(dev,"uses_real_end_fields",False) and hasattr(dev,"z_grid"):
        md=dict(getattr(dev,"metadata",{}) or {})
        z_full=np.linspace(
            float(md.get("tracking_z_start_m",dev.z_grid[0])),
            float(md.get("tracking_z_end_m",dev.z_grid[-1])),
            max(513,int(nper)*96+1),
        )
        char=md.get("characterization_z_range_m")
        if isinstance(char,(list,tuple)) and len(char)==2:
            zc0,zc1=map(float,char)
        else:
            mid=0.5*(z_full[0]+z_full[-1])
            half=0.5*min(3,int(nper))*float(dev.lambda_u)
            zc0,zc1=mid-half,mid+half
    else:
        L=float(nper)*float(dev.lambda_u)
        z_full=np.linspace(0.0,L,max(513,int(nper)*96+1))
        mid=0.5*L
        half=0.5*min(3,int(nper))*float(dev.lambda_u)
        zc0,zc1=mid-half,mid+half

    pts=np.column_stack([np.zeros_like(z_full),np.zeros_like(z_full),z_full])
    B=np.asarray(dev.B(pts),dtype=float)
    bx,by,bz=B[:,0],B[:,1],B[:,2]

    zc=np.linspace(zc0,zc1,max(257,int(round((zc1-zc0)/dev.lambda_u*128))+1))
    Bc=np.asarray(dev.B(np.column_stack([np.zeros_like(zc),np.zeros_like(zc),zc])),dtype=float)
    bxc,byc=Bc[:,0],Bc[:,1]

    ku=2*np.pi/float(dev.lambda_u)
    def amp(signal,n):
        return _harmonic_amplitude(zc,signal,ku,n)
    def vec_amp(n):
        return float(np.hypot(amp(bxc,n),amp(byc,n)))
    a1,a3,a5=vec_amp(1),vec_amp(3),vec_amp(5)

    i1x=float(np.trapezoid(bx,z_full)); i1y=float(np.trapezoid(by,z_full))
    i2x=float(np.trapezoid(_cumtrap(bx,z_full),z_full))
    i2y=float(np.trapezoid(_cumtrap(by,z_full),z_full))

    return {
        "central_z_start_m":float(zc0),"central_z_end_m":float(zc1),
        "full_z_start_m":float(z_full[0]),"full_z_end_m":float(z_full[-1]),
        "Bx_fundamental_T":amp(bxc,1),"By_fundamental_T":amp(byc,1),
        "Btrans_peak_central_T":float(np.max(np.hypot(bxc,byc))),
        "Btrans_peak_global_T":float(np.max(np.hypot(bx,by))),
        "Bz_peak_global_T":float(np.max(np.abs(bz))),
        "first_integral_Bx_Tm":i1x,"first_integral_By_Tm":i1y,
        "second_integral_Bx_Tm2":i2x,"second_integral_By_Tm2":i2y,
        "magnetic_H3_over_H1":float(a3/a1) if a1>0 else np.nan,
        "magnetic_H5_over_H1":float(a5/a1) if a1>0 else np.nan,
    }
