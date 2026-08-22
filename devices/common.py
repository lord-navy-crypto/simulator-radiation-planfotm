from __future__ import annotations
import math
from radia_support import create_linear_ndfeb, make_block
from errors.model import ErrorContext

def material_for(rad, p, br_scale=1.0):
    if p.get("material_mode","Fixed remanence") == "Linear NdFeB + relaxation":
        return create_linear_ndfeb(
            rad,
            br_t=float(p["br_t"])*float(br_scale),
            mu_parallel=float(p.get("mu_parallel",1.05)),
            mu_perpendicular=float(p.get("mu_perpendicular",1.05)),
        )
    return None

def instantiate_block(
    rad,p,ctx,center,size,axis,*,row,index,br_scale=1.0,
    bank_group=0,upper_lower_group=0
):
    ideal_center=[float(v) for v in center]
    ideal_axis=[float(v) for v in axis]
    actual_center,actual_axis,actual_scale,err=ctx.apply(
        ideal_center,ideal_axis,br_scale,
        bank_group=bank_group,upper_lower_group=upper_lower_group
    )
    mat=material_for(rad,p,actual_scale)
    obj=make_block(
        rad,actual_center,size,actual_axis,
        br_t=float(p["br_t"])*actual_scale,
        material=mat,
        segmentation=p.get("segmentation",(1,1,1)),
    )
    meta={
        "row":row,"index":int(index),
        "center":actual_center,"ideal_center":ideal_center,
        "size":[float(v) for v in size],
        "axis":actual_axis,"ideal_axis":ideal_axis,
        "br_scale":float(actual_scale),
        "errors":err,
    }
    return obj,meta

def halbach_axis(inward_normal, phase_rad):
    nx,ny,nz=inward_normal
    c=math.cos(phase_rad); s=math.sin(phase_rad)
    return [c*nx,c*ny,c*nz+s]

def model_container(rad, objects, blocks, kind, material_mode):
    if not objects:
        raise ValueError(f"{kind} generated no magnetic blocks.")
    return {
        "obj":rad.ObjCnt(objects),
        "blocks":blocks,
        "kind":kind,
        "material_mode":material_mode,
    }
