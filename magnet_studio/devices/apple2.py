from __future__ import annotations
import math
from .common import instantiate_block, halbach_axis, model_container
from magnet_studio.errors.model import ErrorContext

def build_apple2(rad,p):
    """
    Physics-informed four-array APPLE-II prototype.

    Four physical magnetic quadrants surround the beam. Polarization control is
    represented by real longitudinal row displacement. This is intentionally
    labelled prototype geometry: it is not a certified replica of a particular
    facility/manufacturer device.
    """
    period=float(p["period_mm"]); periods=int(p["periods"]); gap=float(p["gap_mm"])
    width=float(p["block_width_mm"]); height=float(p["block_height_mm"])
    bpp=max(4,int(p.get("blocks_per_period",4)))
    fill=float(p.get("longitudinal_fill",0.90))
    phase_deg=float(p.get("apple_phase_deg",90.0))
    shift_mode=p.get("apple_shift_mode","Antiparallel")
    dz=period/bpp; block_len=dz*fill; total_len=periods*period
    off=gap/2+height/2
    delta=(phase_deg/360.0)*period

    if shift_mode=="Parallel":
        shifts=[+delta/2,+delta/2,-delta/2,-delta/2]
    else:
        shifts=[+delta/2,-delta/2,+delta/2,-delta/2]

    arrays=[
        ("TR",[+off,+off],shifts[0],+1,+1),
        ("TL",[-off,+off],shifts[1],-1,+1),
        ("BL",[-off,-off],shifts[2],+1,-1),
        ("BR",[+off,-off],shifts[3],-1,-1),
    ]
    ctx=ErrorContext(p); objects=[]; meta=[]; nblocks=periods*bpp
    for row,xy,zshift,bank_group,ul_group in arrays:
        x,y=xy; nrm=math.hypot(x,y); inward=[-x/nrm,-y/nrm,0.0]
        phase0=0.0 if row in ("TR","BL") else math.pi
        for j in range(nblocks):
            z=-total_len/2+(j+0.5)*dz+zshift
            phase=2*math.pi*(j%bpp)/bpp+phase0
            axis=halbach_axis(inward,phase)
            obj,m=instantiate_block(
                rad,p,ctx,[x,y,z],[width,height,block_len],axis,
                row=row,index=j,bank_group=bank_group,upper_lower_group=ul_group
            )
            m["z_shift_mm"]=float(zshift)
            objects.append(obj); meta.append(m)
    return model_container(rad,objects,meta,"APPLE-II",p["material_mode"])
