import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from scipy.special import genlaguerre, sph_harm_y
from skimage import measure


st.set_page_config(
    page_title="Hydrogen Orbital Visualizer",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Interactive Hydrogen Orbital Visualizer")
st.write(
    """
Explore the quantum mechanical orbitals of the hydrogen atom by selecting
different quantum numbers.

The visualization is generated from the analytical solution of the
Schrödinger Equation.
"""
)



st.sidebar.header("Orbital Parameters")

n = st.sidebar.selectbox(
    "Principal Quantum Number (n)",
    [1, 2, 3, 4, 5],
    index=2
)

l = st.sidebar.selectbox(
    "Angular Quantum Number (l)",
    list(range(n)),
    index=min(2, n-1)
)

m = st.sidebar.selectbox(
    "Magnetic Quantum Number (m)",
    list(range(-l, l + 1)),
    index=l
)

grid_resolution = st.sidebar.slider(
    "Grid Resolution",
    40,
    120,
    100
)

grid_extent = st.sidebar.slider(
    "Grid Extent",
    8.0,
    20.0,
    15.0
)

iso_percent = st.sidebar.slider(
    "Isosurface %",
    0.001,
    0.10,
    0.01
)

opacity = st.sidebar.slider(
    "Opacity",
    0.2,
    1.0,
    0.9
)



def hydrogen_wavefunction(n, l, m, X, Y, Z):

    R_grid = np.sqrt(X**2 + Y**2 + Z**2)
    R_grid = np.where(R_grid == 0, 1e-10, R_grid)

    Theta_grid = np.arccos(Z / R_grid)
    Phi_grid = np.arctan2(Y, X)

    a0 = 1.0

    rho = (2 * R_grid) / (n * a0)

    laguerre = genlaguerre(
        n - l - 1,
        2 * l + 1
    )

    normalization = np.sqrt(
        (2 / (n * a0)) ** 3
        * math.factorial(n - l - 1)
        / (2 * n * math.factorial(n + l))
    )

    radial = (
        normalization
        * np.exp(-rho / 2)
        * rho ** l
        * laguerre(rho)
    )

    angular = sph_harm_y(
        l,
        m,
        Theta_grid,
        Phi_grid
    )

    return radial * angular


if st.button("Generate Orbital"):

    with st.spinner("Calculating Wavefunction..."):

        x = np.linspace(
            -grid_extent,
            grid_extent,
            grid_resolution
        )

        y = np.linspace(
            -grid_extent,
            grid_extent,
            grid_resolution
        )

        z = np.linspace(
            -grid_extent,
            grid_extent,
            grid_resolution
        )

        X, Y, Z = np.meshgrid(
            x,
            y,
            z,
            indexing="ij"
        )

        psi = hydrogen_wavefunction(
            n,
            l,
            m,
            X,
            Y,
            Z
        )

        probability = np.abs(psi) ** 2

        iso_value = probability.max() * iso_percent

        verts, faces, normals, values = measure.marching_cubes(
            probability,
            iso_value
        )

        verts_scaled = (
            verts
            * (2 * grid_extent / (grid_resolution - 1))
            - grid_extent
        )

        indices = np.clip(
            np.round(verts).astype(int),
            0,
            grid_resolution - 1
        )

        wave_signs = np.real(
            psi[
                indices[:, 0],
                indices[:, 1],
                indices[:, 2]
            ]
        )

        fig = go.Figure()

        fig.add_trace(
            go.Mesh3d(
                x=verts_scaled[:, 0],
                y=verts_scaled[:, 1],
                z=verts_scaled[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=wave_signs,
                colorscale="RdBu",
                opacity=opacity,
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    specular=0.5,
                    roughness=0.1,
                ),
                colorbar=dict(
                    title="Wave Phase"
                ),
            )
        )

        fig.update_layout(
            title=f"Hydrogen Orbital (n={n}, l={l}, m={m})",
            scene=dict(
                aspectmode="data",
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            margin=dict(
                l=0,
                r=0,
                t=40,
                b=0,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


st.markdown("---")

st.markdown("## About")

st.write("""
This application visualizes the **quantum mechanical orbitals of the hydrogen atom**
by numerically evaluating the analytical solution of the **time-independent Schrödinger equation**.

The hydrogen wavefunction is represented as the product of a radial component and an angular component:
""")

st.latex(r"\psi_{n,l,m}(r,\theta,\phi)=R_{n,l}(r)\,Y_l^m(\theta,\phi)")

st.write("""
where:

- **Rₙₗ(r)** represents the radial wavefunction, determining how the probability changes with distance from the nucleus.
- **Yₗᵐ(θ, φ)** represents the spherical harmonic, determining the orbital's shape and orientation.

The probability of finding the electron at any point in space is calculated using the Born Rule:
""")

st.latex(r"P(r,\theta,\phi)=|\psi_{n,l,m}(r,\theta,\phi)|^2")

st.write("""
The probability density is evaluated over a three-dimensional grid. An **isosurface** is then extracted using the **Marching Cubes algorithm**, producing the familiar 3D orbital shapes. **The Marching Cubes** algorithm is a computer graphics technique. This was published by Lorensen and Cline in 1987. It extracts a polygonal 3D mesh (usually it's triangles) of an isosurface(a surface where every point has an equal scalar value) from a discrete 3D scalar field, such as CT/MRI scan data or 3D Noise. The resulting mesh is rendered interactively with Plotly and colored according to the sign (phase) of the wavefunction.

### Technologies Used

- Python
- NumPy
- SciPy
- Scikit-Image
- Plotly
- Streamlit
""")