import numpy as np

def apply_boundary(U, bc_type="dirichlet"):
    """
    In-place-ish boundary handling for 2D arrays U.
    - dirichlet: u=0 on boundary
    - neumann  : zero normal gradient (copy inner neighbor)
    - periodic : wrap
    """
    V = U.copy()
    if bc_type == "dirichlet":
        V[0, :] = 0; V[-1, :] = 0; V[:, 0] = 0; V[:, -1] = 0
    elif bc_type == "neumann":
        V[0, :] = V[1, :]; V[-1, :] = V[-2, :]
        V[:, 0] = V[:, 1]; V[:, -1] = V[:, -2]
    elif bc_type == "periodic":
        V[0, :]   = V[-2, :]
        V[-1, :]  = V[1,  :]
        V[:, 0]   = V[:, -2]
        V[:, -1]  = V[:, 1]
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")
    return V

def cfl_dt(c, dx, dy):
    # 2D Cartesian explicit scheme stability (CFL):
    # c*dt/min(dx,dy) < 1/sqrt(2)
    return 0.45 * min(dx, dy) / (c * np.sqrt(2.0))
