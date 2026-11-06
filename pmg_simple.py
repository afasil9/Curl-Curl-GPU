#%%
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ufl
from dolfinx import fem, la, mesh
from dolfinx.fem import assemble_matrix
from mpi4py import MPI
from basix import CellType, ElementFamily, LagrangeVariant

from basic_operators import (
    curlcurl_reference_matrix,   # (element, quadrature_degree) -> (12, nd, nd)
    permutation_classes,         # (mesh) -> (clas, uperms)
    dense_transform,             # (V, which, uperms) -> (n_classes, nd, nd)
    piola_map,                   # (mesh) -> (n_cells, 12)
    fold_interpolation,          # (V_src, V_dst, e_src, e_dst, uperms)
    fold_discrete_gradient,      # (V_lag, V_ned, e_lag, e_ned, uperms)
    boundary_dof_mask,           # (V) -> 0/1 array
    apply_transfer,              # (x, V_src, V_dst, B_folded, clas)
    apply_transfer_transpose,    # (x, V_src, V_dst, B_folded, clas)
    CurlCurlMatFree,             # (V, folded_ref, cell_coef, clas, dofmap)
)
import basix
import basix.ufl




def build_level(msh, degree, clas, uperms, cell_coef):
    """Gives the action of the operator on each level (polynomial order `degree`).

    Returns a dict: the masked operator plus everything its smoother needs.
    `clas`, `uperms` and `cell_coef` are mesh properties, so they are computed
    once outside and shared by every level.
    """
    elem = basix.create_element(ElementFamily.N1E, CellType.tetrahedron, degree,
                                lagrange_variant=LagrangeVariant.legendre)
    V = fem.functionspace(msh, basix.ufl.wrap_element(elem))
    dofmap = V.dofmap.list

    # Reference matrices on the reference tet: 12 of them, one per Piola coefficient.
    refs = curlcurl_reference_matrix(elem, quadrature_degree=2 * degree + 2)

    # Orientation transforms, one per permutation class (not per cell).
    T  = dense_transform(V, "T",  uperms)
    Tt = dense_transform(V, "Tt", uperms)

    # Fold the transforms into the reference matrices once: T R_k T^t.
    folded_ref = np.einsum("pai,kij,pjb->pkab", T, refs, Tt, optimize=True)

    A_action = CurlCurlMatFree(V, folded_ref, cell_coef, clas, dofmap)
    mask = 1.0 - boundary_dof_mask(V)[:A_action.ndofs]

    lvl = {"degree": degree, "element": elem, "V": V, "A_action": A_action,
           "ndofs": A_action.ndofs, "mask": mask, "lam_max": None}

    # Order matters: diag() needs mask, and estimate_lambda_max needs d_inv.
    lvl["d_inv"] = 1.0 / diag(lvl)
    lvl["lam_max"] = estimate_lambda_max(lvl)

    print(f"  degree {degree}: ndofs={lvl['ndofs']:6d}  "
          f"dofs/cell={dofmap.shape[1]:3d}  bc={int((1.0 - mask).sum()):5d}  "
          f"folded_ref={folded_ref.shape}  lam_max={lvl['lam_max']:.4f}")
    return lvl


def A(lvl, x):
    """Masked action: y = M A M x + (I - M) x."""
    m = lvl["mask"]
    return m * lvl["A_action"].matvec(m * x) + (1.0 - m) * x


def diag(lvl):
    """Masked diagonal: 1.0 on constrained dofs."""
    m = lvl["mask"]
    return m * lvl["A_action"].diagonal() + (1.0 - m)



def estimate_lambda_max(lvl, n_it=20, seed=0):
    """Power iteration on D^-1 A.  Masked start, or it locks onto the
    constrained block's eigenvalue of 1 and underestimates."""
    v = lvl["mask"] * np.random.default_rng(seed).standard_normal(lvl["ndofs"])
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(n_it):
        w = lvl["d_inv"] * A(lvl, v)
        lam = np.linalg.norm(w)
        v = w / lam
    return lam


def jacobi(lvl, b, x, steps=2, lower=0.3):
    """Damped Jacobi: x <- x + omega D^-1 (b - A x), omega = 2/(a+c)."""
    omega = 2.0 / (lower * lvl["lam_max"] + 1.1 * lvl["lam_max"])
    d_inv = lvl["d_inv"]
    for i in range(steps):
        x = x + omega * d_inv * (b - A(lvl, x))
    return x


def chebyshev(lvl, b, x, steps=2, lower=0.3):
    """Chebyshev smoother on [lower*lam_max, 1.1*lam_max]."""
    a, c = lower * lvl["lam_max"], 1.1 * lvl["lam_max"]
    theta, delta = 0.5 * (c + a), 0.5 * (c - a)
    sigma = theta / delta
    d_inv = lvl["d_inv"]

    r   = b - A(lvl, x)
    rho = 1.0 / sigma
    d   = (r * d_inv) / theta
    x   = x + d

    # Power iteration
    for i in range(1, steps):
        r = b - A(lvl, x)
        rho_new = 1.0 / (2.0 * sigma - rho)
        d = rho * rho_new * d + (2.0 * rho_new / delta) * (r * d_inv)
        x = x + d
        rho = rho_new

    return x


def coarse_matrix(lvl):

    V = lvl["V"]
    msh = V.mesh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    bc = fem.dirichletbc(fem.Function(V),
                         fem.locate_dofs_topological(V, tdim - 1, facets))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form((ufl.inner(ufl.curl(u), ufl.curl(v)) + ufl.inner(u, v)) * ufl.dx)

    A_csr = fem.assemble_matrix(a, bcs=[bc], diag=1.0)
    A_csr.scatter_reverse()

    nd = lvl["ndofs"] 
    A_sp = A_csr.to_scipy().tocsr()[:nd, :nd].tocsc()
    A_sp.eliminate_zeros()  
    return A_sp


def coarse_lu(lvl):
    """
    Coarse Direct solve
    """
    return spla.splu(coarse_matrix(lvl))

def exact_solution(msh):
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    return ufl.as_vector((ufl.sin(pi * x[1]) * ufl.sin(pi * x[2]),
                          ufl.sin(pi * x[0]) * ufl.sin(pi * x[2]),
                          ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])))


def manufactured_rhs(lvl, u_ex):
    v = ufl.TestFunction(lvl["V"])
    f = ufl.curl(ufl.curl(u_ex)) + u_ex
    b = fem.assemble_vector(fem.form(ufl.inner(f, v) * ufl.dx))
    b.scatter_reverse(la.InsertMode.add)
    return lvl["mask"] * b.array[:lvl["ndofs"]]


n = 8
degree_ladder = [1, 2, 3]
msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
cell_coef = piola_map(msh)

clas, uperms = permutation_classes(msh)

print(f"building levels {degree_ladder} on a {n}x{n}x{n} unit cube")
levels = [build_level(msh, d, clas, uperms, cell_coef) for d in degree_ladder]


fine_lvl = len(levels) - 1

u_ex   = exact_solution(msh)
b      = manufactured_rhs(levels[fine_lvl], u_ex)

#Initial guess 0
x = np.zeros_like(b)

# Applying smoother on fine mesh
x = chebyshev(levels[fine_lvl], b, np.zeros_like(b), steps =5)

r = b - A(levels[fine_lvl], x)
transfers = [
    fold_interpolation(levels[i]["V"], levels[i+1]["V"],
                       levels[i]["element"], levels[i+1]["element"], uperms)
    for i in range(len(levels) - 1)
]

r_c = levels[fine_lvl-1]["mask"] * apply_transfer_transpose(
        levels[fine_lvl]["mask"] * r,
        levels[fine_lvl - 1]["V"],    # V_src = coarse
        levels[fine_lvl]["V"],                # V_dst = fine
        transfers[fine_lvl - 1],
        clas)


coarse_lvl = fine_lvl - 1

# A e_c = r_c
e_c = chebyshev(levels[coarse_lvl], r_c, np.zeros_like(r_c), steps=5)

r_remaining = r_c - A(levels[coarse_lvl], e_c)
low_lvl = coarse_lvl - 1

r_cc = levels[low_lvl]["mask"] * apply_transfer_transpose(
        levels[coarse_lvl]["mask"] * r_remaining,
        levels[low_lvl]["V"],       # V_src = coarse 
        levels[coarse_lvl]["V"],    # V_dst = fine   
        transfers[low_lvl],
        clas)


# Bottom of the ladder: solve exactly instead of smoothing.
lu = coarse_lu(levels[low_lvl])
e_0 = lu.solve(r_cc)

print("coarse solve residual:",
      np.linalg.norm(A(levels[low_lvl], e_0) - r_cc) / np.linalg.norm(r_cc))

