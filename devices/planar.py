from __future__ import annotations
import math
from .common import instantiate_block, halbach_axis, model_container
from errors.model import ErrorContext

def build_planar(rad,p):
    period=float(p["period_mm"]); periods=int(p["periods"]); gap=float(p["gap_mm"])
    width=float(p["block_width_mm"]); height=float(p["block_height_mm"])
    bpp=int(p.get("blocks_per_period",4))
    if bpp<4: raise ValueError("Planar model requires at least 4 blocks per period.")
    fill=float(p.get("longitudinal_fill",0.92))
    dz=period/bpp; block_len=dz*fill; total_len=periods*period
    y0=gap/2+height/2
    ctx=ErrorContext(p)
    objects=[]; meta=[]
    rows=[
        ("top",[0,-1,0],+y0,+1.0,+1,+1),
        ("bottom",[0,+1,0],-y0,-1.0,-1,-1),
    ]
    nblocks=periods*bpp
    for row,inward,y,phase_sign,bank_group,ul_group in rows:
        for j in range(nblocks):
            z=-total_len/2+(j+0.5)*dz
            phase=phase_sign*2*math.pi*(j%bpp)/bpp
            axis=halbach_axis(inward,phase)
            obj,m=instantiate_block(
                rad,p,ctx,[0,y,z],[width,height,block_len],axis,
                row=row,index=j,bank_group=bank_group,upper_lower_group=ul_group
            )
            objects.append(obj); meta.append(m)
    return model_container(rad,objects,meta,"Planar",p["material_mode"])
