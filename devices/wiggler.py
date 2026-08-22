from .planar import build_planar

def build_wiggler(rad,p):
    model=build_planar(rad,p)
    model["kind"]="Wiggler"
    model["topology_note"]="Planar Halbach topology; wiggler operation is determined by the resulting large K."
    return model
