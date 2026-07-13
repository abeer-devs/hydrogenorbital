import math
import numpy as np
import plotly.graph_objects as go
from scipy.special import genlaguerre, sph_harm_y
from skimage import measure


def hydrogen_wavefunction(n, l, m, X, Y, Z):
    """Calculates the complex wavefunction amplitude at grid coordinates X, Y,

    Z.
    """
    R_grid = np.sqrt(X**2 + Y**2 + Z**2)
    R_grid = np.where(R_grid == 0, 1e-10, R_grid) 

    Theta_grid = np.arccos(Z / R_grid)
    Phi_grid = np.arctan2(Y, X)

    a_0 = 1.0
    rho = (2 * R_grid) / (n * a_0)
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)

    normalization_r = np.sqrt(
        (2 / (n * a_0)) ** 3
        * math.factorial(n - l - 1)
        / (2 * n * math.factorial(n + l))
    )
    radial = normalization_r * np.exp(-rho / 2) * (rho**l) * laguerre(rho)

    angular = sph_harm_y(l, m, Theta_grid, Phi_grid)

    return radial * angular



n, l, m = 3, 2, 0  # 3d orbital configuration

grid_extent = 15.0
grid_resolution = 100

x = np.linspace(-grid_extent, grid_extent, grid_resolution)
y = np.linspace(-grid_extent, grid_extent, grid_resolution)
z = np.linspace(-grid_extent, grid_extent, grid_resolution)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

psi = hydrogen_wavefunction(n, l, m, X, Y, Z)
prob_density = np.abs(psi) ** 2


iso_value = prob_density.max() * 0.01

try:
    verts, faces, normals, values = measure.marching_cubes(
        prob_density, iso_value
    )
except RuntimeError as e:
    print(
        f"Error: {e}\nTry adjusting grid_extent or choosing a lower iso_value percentage."
    )
    raise

# Map index-based vertices back into real coordinate bounds
verts_scaled = verts * (2 * grid_extent / (grid_resolution - 1)) - grid_extent

verts_indices = np.round(verts).astype(int)
verts_indices = np.clip(verts_indices, 0, grid_resolution - 1)
wave_signs = np.real(
    psi[verts_indices[:, 0], verts_indices[:, 1], verts_indices[:, 2]]
)

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=verts_scaled[:, 0],
            y=verts_scaled[:, 1],
            z=verts_scaled[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=wave_signs,  # Color surface by quantum phase sign (+/-)
            colorscale="RdBu",
            opacity=0.9,
            lighting=dict(
                ambient=0.4, diffuse=0.8, specular=0.5, roughness=0.1
            ),
            colorbar=dict(title="Wave Phase Sign"),
        )
    ]
)

fig.update_layout(
    title=f"Interactive 3D Hydrogen Orbital Model (n={n}, l={l}, m={m})",
    scene=dict(
        xaxis_title="X (Bohr radii)",
        yaxis_title="Y (Bohr radii)",
        zaxis_title="Z (Bohr radii)",
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

fig.show()
