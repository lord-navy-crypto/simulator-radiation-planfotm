from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

def field_lines(z_mm, B):
    fig=go.Figure()
    fig.add_scatter(x=z_mm,y=B[:,0],name="Bx")
    fig.add_scatter(x=z_mm,y=B[:,1],name="By")
    fig.add_scatter(x=z_mm,y=B[:,2],name="Bz")
    fig.update_layout(xaxis_title="z (mm)",yaxis_title="B (T)",title="On-axis magnetic field")
    return fig

def slice_heatmap(axis_values_mm, z_mm, Bslice, component_index=1, transverse_label="x"):
    data=Bslice[:,:,component_index]
    fig=go.Figure(go.Heatmap(x=axis_values_mm,y=z_mm,z=data))
    comp=["Bx","By","Bz"][component_index]
    fig.update_layout(
        title=f"{comp} {transverse_label.upper()}Z slice",
        xaxis_title=f"{transverse_label} (mm)",
        yaxis_title="z (mm)"
    )
    return fig

def field_cones(x_mm,y_mm,z_mm,B3):
    X,Y,Z=np.meshgrid(x_mm,y_mm,z_mm,indexing="xy")
    # B3 is [z,y,x,component], transpose to match mesh flattened order.
    U=np.transpose(B3[:,:,:,0],(1,2,0))
    V=np.transpose(B3[:,:,:,1],(1,2,0))
    W=np.transpose(B3[:,:,:,2],(1,2,0))
    fig=go.Figure(go.Cone(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
        u=U.ravel(), v=V.ravel(), w=W.ravel(),
        sizemode="absolute", sizeref=max(float(np.max(np.linalg.norm(B3,axis=3))),1e-6),
        anchor="tail"
    ))
    fig.update_layout(
        title="3D magnetic field map",
        scene=dict(xaxis_title="x (mm)",yaxis_title="y (mm)",zaxis_title="z (mm)")
    )
    return fig

def trajectory_plot(z_mm,tr):
    fig=go.Figure()
    fig.add_scatter(x=z_mm,y=tr["x_mm"],name="x(z)")
    fig.add_scatter(x=z_mm,y=tr["y_mm"],name="y(z)")
    fig.update_layout(xaxis_title="z (mm)",yaxis_title="Transverse displacement (mm)",title="Electron trajectory (small-angle)")
    return fig


def _cuboid_vertices_faces(center,size,offset=0):
    cx,cy,cz=center; sx,sy,sz=[v/2 for v in size]
    verts=[
        (cx-sx,cy-sy,cz-sz),(cx+sx,cy-sy,cz-sz),
        (cx+sx,cy+sy,cz-sz),(cx-sx,cy+sy,cz-sz),
        (cx-sx,cy-sy,cz+sz),(cx+sx,cy-sy,cz+sz),
        (cx+sx,cy+sy,cz+sz),(cx-sx,cy+sy,cz+sz),
    ]
    faces=[
        (0,1,2),(0,2,3),(4,6,5),(4,7,6),
        (0,4,5),(0,5,1),(1,5,6),(1,6,2),
        (2,6,7),(2,7,3),(3,7,4),(3,4,0)
    ]
    return verts,[(a+offset,b+offset,c+offset) for a,b,c in faces]

def geometry_view(blocks,max_blocks=600,show_axes=True):
    import plotly.graph_objects as go
    if not blocks:
        return go.Figure()
    # Evenly subsample long devices so geometry remains interactive.
    if len(blocks)>max_blocks:
        idx=np.linspace(0,len(blocks)-1,max_blocks,dtype=int)
        selected=[blocks[i] for i in idx]
    else:
        selected=list(blocks)

    by_row={}
    for b in selected:
        by_row.setdefault(str(b.get("row","row")),[]).append(b)

    fig=go.Figure()
    for row,items in by_row.items():
        verts=[]; faces=[]
        for b in items:
            v,f=_cuboid_vertices_faces(b["center"],b["size"],len(verts))
            verts.extend(v); faces.extend(f)
        x=[v[0] for v in verts]; y=[v[1] for v in verts]; z=[v[2] for v in verts]
        i=[f[0] for f in faces]; j=[f[1] for f in faces]; k=[f[2] for f in faces]
        fig.add_trace(go.Mesh3d(
            x=x,y=y,z=z,i=i,j=j,k=k,
            name=row,opacity=0.55,flatshading=True,showscale=False
        ))

    if show_axes:
        centers=np.asarray([b["center"] for b in selected],float)
        axes=np.asarray([b["axis"] for b in selected],float)
        sizes=np.asarray([b["size"] for b in selected],float)
        scale=float(np.median(np.min(sizes,axis=1))*0.6) if len(sizes) else 1.0
        fig.add_trace(go.Cone(
            x=centers[:,0],y=centers[:,1],z=centers[:,2],
            u=axes[:,0]*scale,v=axes[:,1]*scale,w=axes[:,2]*scale,
            name="Magnetization direction",anchor="tail",
            sizemode="absolute",sizeref=max(scale,1e-6),showscale=False
        ))
    fig.update_layout(
        title=f"3D magnet geometry ({len(selected)} of {len(blocks)} blocks shown)",
        scene=dict(xaxis_title="x (mm)",yaxis_title="y (mm)",zaxis_title="z (mm)",aspectmode="data")
    )
    return fig

def ideal_error_field_plot(z_mm,Bideal,Berr):
    import plotly.graph_objects as go
    fig=go.Figure()
    fig.add_scatter(x=z_mm,y=Bideal[:,0],name="Ideal Bx")
    fig.add_scatter(x=z_mm,y=Berr[:,0],name="Error Bx")
    fig.add_scatter(x=z_mm,y=Bideal[:,1],name="Ideal By")
    fig.add_scatter(x=z_mm,y=Berr[:,1],name="Error By")
    fig.update_layout(title="Ideal vs error-model on-axis field",xaxis_title="z (mm)",yaxis_title="B (T)")
    return fig


def electron_phase_plot(phase_result):
    import plotly.graph_objects as go
    fig = go.Figure()
    z = phase_result.get("positions_mm", [])
    p = phase_result.get("phase_error_deg", [])
    fig.add_scatter(x=z, y=p, mode="lines+markers", name="Electron phase error")
    fig.add_hline(y=0.0)
    fig.update_layout(
        title="Trajectory-derived electron phase error",
        xaxis_title="z (mm)",
        yaxis_title="Phase error (deg)"
    )
    return fig
