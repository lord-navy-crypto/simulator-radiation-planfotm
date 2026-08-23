from __future__ import annotations
import math
from .common import instantiate_block, halbach_axis, model_container
from magnet_studio.errors.model import ErrorContext

def build_elliptical(rad,p):
    period=float(p["period_mm"]); periods=int(p["periods"]); gap=float(p["gap_mm"])
    width=float(p["block_width_mm"]); height=float(p["block_height_mm"])
    bpp=max(4,int(p.get("blocks_per_period",8)))
    fill=float(p.get("longitudinal_fill",0.90))
    e=min(1.0,max(0.0,float(p.get("ellipticity",0.5))))
    dz=period/bpp; block_len=dz*fill; total_len=periods*period
    r=gap/2+height/2; ctx=ErrorContext(p)
    rows=[
        ("top",[0,-1,0],[0,+r],0.0,1.0,+1,+1),
        ("bottom",[0,+1,0],[0,-r],math.pi,1.0,-1,-1),
        ("right",[-1,0,0],[+r,0],math.pi/2,e,+1,0),
        ("left",[+1,0,0],[-r,0],3*math.pi/2,e,-1,0),
    ]
    objects=[]; meta=[]; nblocks=periods*bpp
    for row,inward,xy,phase0,scale,bank_group,ul_group in rows:
        if scale<=1e-12: continue
        x,y=xy
        for j in range(nblocks):
            z=-total_len/2+(j+0.5)*dz
            phase=2*math.pi*(j%bpp)/bpp+phase0
            axis=halbach_axis(inward,phase)
            obj,m=instantiate_block(
                rad,p,ctx,[x,y,z],[width,height,block_len],axis,
                row=row,index=j,br_scale=scale,
                bank_group=bank_group,upper_lower_group=ul_group
            )
            objects.append(obj); meta.append(m)
    return model_container(rad,objects,meta,"Elliptical",p["material_mode"])
