import plotly.graph_objects as go
import trimesh
import numpy as np
from plotly.subplots import make_subplots
import math

# Load the 3D model
sphere_mesh = trimesh.load('IcoSphere_0.4.obj')
#x_arrow_mesh = trimesh.load('X_Arrow.obj')
#y_arrow_mesh = trimesh.load('Y_Arrow.obj')
#z_arrow_mesh = trimesh.load('Z_Arrow.obj')

def get_protein_traces(sequence, protein,
                       min_x=None, max_x=None, 
                       min_y=None, max_y=None, 
                       min_z=None, max_z=None):
    """
    Return the Plotly traces and a scene dictionary (for axis ranges, aspect mode, etc.)
    without creating a final figure.
    """
    molecule_colors = ["green", "orange", "blue"]
    mesh_vertex_count = sphere_mesh.vertices.shape[0]
    mesh_face_count   = sphere_mesh.faces.shape[0]

    traces = []
    # Build Mesh3d traces
    for i, color in enumerate(molecule_colors):
        if not (sequence == i).any():
            continue
        
        molecules = protein[sequence == i]

        vertices = (sphere_mesh.vertices[None, :, :] + molecules[:, None, :]).reshape(-1, 3)
        repeated_indices = np.tile(sphere_mesh.faces, (molecules.shape[0], 1))
        index_offsets = np.repeat(np.arange(molecules.shape[0]) * mesh_vertex_count, mesh_face_count)
        indices = repeated_indices + index_offsets[:, None]

        traces.append(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=indices[:, 0],
            j=indices[:, 1],
            k=indices[:, 2],
            color=color,
            hoverinfo='skip'
        ))

    # Add the backbone (lines) trace
    traces.append(go.Scatter3d(
        x=protein[:, 0],
        y=protein[:, 1],
        z=protein[:, 2],
        mode='lines',
        line=dict(color='black', width=10)
    ))

    # Build the scene config (axis ranges, aspect ratios, etc.)
    if min_x is not None:
        # If we have bounding box info
        scene_dict = dict(
            aspectmode="manual",
            aspectratio={
                'x': max_x - min_x, 
                'y': max_y - min_y, 
                'z': max_z - min_z
            },
            xaxis=dict(range=[min_x, max_x], autorange=False),
            yaxis=dict(range=[min_y, max_y], autorange=False),
            zaxis=dict(range=[min_z, max_z], autorange=False)
        )
    else:
        # Let Plotly choose the best aspect ratio
        scene_dict = dict(aspectmode="data")

    return traces, scene_dict

def plot_batch(sequence, proteins):

    min_x = np.min(proteins[:, :, 0]) - 0.5
    max_x = np.max(proteins[:, :, 0]) + 0.5
    min_y = np.min(proteins[:, :, 1]) - 0.5
    max_y = np.max(proteins[:, :, 1]) + 0.5
    min_z = np.min(proteins[:, :, 2]) - 0.5
    max_z = np.max(proteins[:, :, 2]) + 0.5

    n = len(proteins)
    s = math.ceil(math.sqrt(n))
    
    # Create the subplots (all 3D)
    fig = make_subplots(
        rows=s, 
        cols=s,
        specs=[[{"type": "scene"} for _ in range(s)] for _ in range(s)],
        vertical_spacing=0.02,  # tweak spacing as needed
        horizontal_spacing=0.02
    )
    
    # We can create just one scene dict if all bounding boxes are the same.
    # If you want each subplot to have its own bounding, generate per protein.
    # For simplicity, we generate one scene_dict from the FIRST protein only
    # (assuming they all share the same bounding box).
    _, shared_scene_dict = get_protein_traces(sequence, proteins[0],
                                              min_x, max_x,
                                              min_y, max_y,
                                              min_z, max_z)

    # We'll loop over each protein, get the traces, and insert them
    for i, protein in enumerate(proteins):
        row = (i // s) + 1
        col = (i % s) + 1
        
        # Get the traces for this particular protein
        traces, _ = get_protein_traces(sequence, protein,
                                       min_x, max_x,
                                       min_y, max_y,
                                       min_z, max_z)

        # Add each trace to the subplot at (row, col)
        for tr in traces:
            fig.add_trace(tr, row=row, col=col)
    
    # Now we must apply the scene layout to each subplot. By default,
    # subplot #1 is named "scene", #2 is "scene2", #3 is "scene3", etc.
    # We'll loop over each subplot index:
    for subplot_idx in range(1, n+1):
        scene_name = "scene" if subplot_idx == 1 else f"scene{subplot_idx}"
        fig.update_layout({scene_name: shared_scene_dict})

    # (Optional) Some final figure layout
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(l=5, r=5, t=5, b=5)
    )
    
    return fig

def plot_protein(sequence, protein, min_x=None, max_x=None, min_y=None, max_y=None, min_z=None, max_z=None):
    
    # Plot the mesh using Plotly
    plotting_data = []
    molecule_colors = ["green", "orange", "blue"]
    
    mesh_vertex_count = sphere_mesh.vertices.shape[0]
    mesh_face_count = sphere_mesh.faces.shape[0]
    for i in range(len(molecule_colors)):
        molecules = protein[sequence == i]

        vertices = (sphere_mesh.vertices[None, :, :] + molecules[:, None, :]).reshape(-1, 3)
        repeated_indices = np.tile(sphere_mesh.faces, (molecules.shape[0], 1))
        index_offsets = np.repeat(np.arange(molecules.shape[0]) * mesh_vertex_count, mesh_face_count)
        indices = repeated_indices + index_offsets[:, None]
        
        plotting_data.append(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=indices[:, 0], j=indices[:, 1], k=indices[:, 2],
            color=molecule_colors[i],
            hoverinfo='skip'
        ))
    
    plotting_data.append(go.Scatter3d(
        x=protein[:,0],
        y=protein[:,1],
        z=protein[:,2],
        mode='lines',
        line=dict(color='black', width=10)
    ))
    
    fig = go.Figure(data=plotting_data)
    
    # Set layout for 3D plot
    scene = None
    if min_x is not None:
        scene=dict(aspectmode="manual",
            aspectratio={'x': max_x-min_x, 'y': max_y-min_y, 'z': max_z-min_z},
            xaxis=dict(range=[min_x, max_x], autorange=False),
            yaxis=dict(range=[min_y, max_y], autorange=False),
            zaxis=dict(range=[min_z, max_z], autorange=False))
    else:
        scene=dict(aspectmode="data")
    
    fig.update_layout(width=500, height=500, uirevision='constant',
        scene=scene,
        margin=dict(l=5, r=5, t=30, b=5)
    )
    return fig


def plot_protein_trajectory(timed_positions, sequence):
    
    from dash import Dash, dcc, html, Input, Output
    import plotly.express as px
    
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.H4(f'Protein trajectory'),
        dcc.Graph(id="graph"),
        dcc.Slider(
            id='slider',
            min=0, max=len(timed_positions) - 1, step=1,
            value=0,
            updatemode='drag'
        )
    ])
    
    @app.callback(
        Output("graph", "figure"), 
        Input("slider", "value"))
    def update_time(slider_value):

        min_x = min([np.min(positions[:,0]) for positions in timed_positions]) - 0.5
        max_x = max([np.max(positions[:,0]) for positions in timed_positions]) + 0.5
        min_y = min([np.min(positions[:,1]) for positions in timed_positions]) - 0.5
        max_y = max([np.max(positions[:,1]) for positions in timed_positions]) + 0.5
        min_z = min([np.min(positions[:,2]) for positions in timed_positions]) - 0.5
        max_z = max([np.max(positions[:,2]) for positions in timed_positions]) + 0.5
        fig = plot_protein(sequence, timed_positions[slider_value], min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, min_z=min_z, max_z=max_z)

        fig.update_layout(title=f"t={slider_value}")
        return fig
    
    app.run_server(debug=True)