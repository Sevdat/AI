# # render.py
# import torch
# import numpy as np
# import plotly.graph_objs as go
# from gnn import data, model  # Import data and model from GNN.py

# # Forward pass to get predictions
# output: torch.Tensor = model(data.x, data.edge_index)

# # Generate random 3D positions for nodes
# num_nodes = data.x.size(0)
# pos = np.random.rand(num_nodes, 3)  # Random 3D positions for visualization

# # Create edges for visualization
# edge_trace = []
# for edge in data.edge_index.t().numpy():
#     x0, y0, z0 = pos[edge[0]]
#     x1, y1, z1 = pos[edge[1]]
#     edge_trace.append(go.Scatter3d(
#         x=[x0, x1], y=[y0, y1], z=[z0, z1],
#         mode='lines',
#         line=dict(width=2, color='#888'),
#         hoverinfo='none'
#     ))

# # Create nodes for visualization
# node_trace = go.Scatter3d(
#     x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
#     mode='markers+text',
#     marker=dict(
#         size=10,
#         color=output.argmax(dim=1).numpy(),  # Color nodes by predicted class
#         colorscale='Viridis',
#         line=dict(color='black', width=0.5)
#     ),
#     text=[f'Node {i}' for i in range(num_nodes)],
#     hoverinfo='text'
# )

# # Create the figure
# fig = go.Figure(data=edge_trace + [node_trace])

# # Set layout to render as a cube
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(
#             title='X',
#             autorange=False,  # Disable auto-scaling
#             range=[0, 1],     # Set fixed range
#             showticklabels=False  # Hide tick labels to reduce distraction
#         ),
#         yaxis=dict(
#             title='Y',
#             autorange=False,  # Disable auto-scaling
#             range=[0, 1],     # Set fixed range
#             showticklabels=False  # Hide tick labels to reduce distraction
#         ),
#         zaxis=dict(
#             title='Z',
#             autorange=False,  # Disable auto-scaling
#             range=[0, 1],     # Set fixed range
#             showticklabels=False  # Hide tick labels to reduce distraction
#         ),
#         aspectmode='cube',  # Lock aspect ratio to render as a cube
#         camera=dict(
#             eye=dict(x=1.5, y=1.5, z=1.5),  # Set a fixed camera position
#             up=dict(x=0, y=0, z=1),         # Lock the z-axis as "up"
#             projection=dict(type='orthographic')  # Use orthographic projection
#         )
#     ),
#     margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
#     showlegend=False,
#     width=500,  # Set the width of the graph
#     height=500  # Set the height of the graph
# )

# # Show the figure
# fig.show()