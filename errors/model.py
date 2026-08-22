from __future__ import annotations
import math
import numpy as np

def _unit(v):
    a=np.asarray(v,dtype=float)
    n=float(np.linalg.norm(a))
    if n <= 0:
        raise ValueError("zero direction vector")
    return a/n

def _rotate(v, axis, angle_rad):
    """Rodrigues rotation."""
    v=_unit(v); k=_unit(axis)
    c=math.cos(angle_rad); s=math.sin(angle_rad)
    return v*c + np.cross(k,v)*s + k*np.dot(k,v)*(1-c)

class ErrorContext:
    """
    Reproducible manufacturing-error model.

    Random parameters are interpreted as 1-sigma values:
      field amplitude error (%)       -> per-block Br scale
      longitudinal position error mm  -> per-block z
      transverse position error mm    -> per-block x/y
      magnetization angle error deg   -> per-block easy-axis rotation

    Systematic parameters:
      gap asymmetry mm                -> shifts magnetic mid-plane relative to beam
      bank strength imbalance (%)     -> opposite bank groups receive +/- half imbalance
    """
    def __init__(self, p):
        self.enabled=bool(p.get("errors_enabled",False))
        self.rng=np.random.default_rng(int(p.get("error_seed",12345)))
        self.field_sigma=float(p.get("field_error_pct",0.0))/100.0
        self.z_sigma=float(p.get("longitudinal_error_mm",0.0))
        self.xy_sigma=float(p.get("transverse_error_mm",0.0))
        self.angle_sigma=math.radians(float(p.get("angle_error_deg",0.0)))
        self.gap_asym=float(p.get("gap_asymmetry_mm",0.0))
        self.bank_imbalance=float(p.get("bank_imbalance_pct",0.0))/100.0

    def apply(self, center, axis, br_scale=1.0, *, bank_group=0, upper_lower_group=0):
        c=np.asarray(center,dtype=float).copy()
        a=_unit(axis)
        scale=float(br_scale)
        record={
            "dBr_rel":0.0,"dx_mm":0.0,"dy_mm":0.0,"dz_mm":0.0,
            "dangle_deg":0.0,"gap_shift_mm":0.0,"bank_scale":1.0,
        }
        if not self.enabled:
            return c.tolist(),a.tolist(),scale,record

        # Random strength.
        if self.field_sigma:
            d=float(self.rng.normal(0.0,self.field_sigma))
            scale*=1.0+d
            record["dBr_rel"]=d

        # Random position.
        if self.xy_sigma:
            dx=float(self.rng.normal(0.0,self.xy_sigma))
            dy=float(self.rng.normal(0.0,self.xy_sigma))
            c[0]+=dx; c[1]+=dy
            record["dx_mm"]=dx; record["dy_mm"]=dy
        if self.z_sigma:
            dz=float(self.rng.normal(0.0,self.z_sigma))
            c[2]+=dz
            record["dz_mm"]=dz

        # Magnetization direction error.
        if self.angle_sigma:
            ang=float(self.rng.normal(0.0,self.angle_sigma))
            raw=self.rng.normal(size=3)
            # Avoid choosing an axis almost parallel to magnetization.
            raw=raw-np.dot(raw,a)*a
            if np.linalg.norm(raw)<1e-12:
                raw=np.array([a[1],-a[0],0.0])
                if np.linalg.norm(raw)<1e-12:
                    raw=np.array([1.0,0.0,0.0])
            a=_rotate(a,raw,ang)
            record["dangle_deg"]=math.degrees(ang)

        # Gap asymmetry: positive value means upper half-gap grows and lower shrinks,
        # shifting both upper/lower magnet centres upward by half the asymmetry.
        if upper_lower_group and self.gap_asym:
            shift=0.5*self.gap_asym
            c[1]+=shift
            record["gap_shift_mm"]=shift

        # Bank imbalance: group +1 receives +delta/2, group -1 receives -delta/2.
        if bank_group and self.bank_imbalance:
            f=1.0+0.5*self.bank_imbalance*float(bank_group)
            scale*=f
            record["bank_scale"]=f

        return c.tolist(),a.tolist(),scale,record
