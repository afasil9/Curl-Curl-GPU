#%%
import basix
import numpy as np
import scipy.sparse as sp
from basix import CellType, ElementFamily, LagrangeVariant
from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
from dolfinx.fem import form, assemble_matrix
from dolfinx.fem.petsc import interpolation_matrix, discrete_gradient
from numpy.linalg import norm

def curlcurl_reference_matrix(element, quadrature_degree=2):

    """Computes reference matrix over a reference tet
    (12, ndofs, ndofs) where 12 is the number of coefficients for the Piola map. The first 6 are for the curl-curl part, the last 6 are for the mass part."""
    pts, wts = basix.make_quadrature(element.cell_type, quadrature_degree)

    tab = element.tabulate(1, pts) #Evaluate the element basis functions and their first derivatives at the quadrature points.
    phi = tab[0] # Undifferentiated basis functions
    dx, dy, dz = tab[1], tab[2], tab[3]
    curl = np.stack(
        [
            dy[..., 2] - dz[..., 1],
            dz[..., 0] - dx[..., 2],
            dx[..., 1] - dy[..., 0],
        ],
        axis=-1,
    )

    mass_matrix = np.einsum("q,qia,qjb->abij", wts, phi, phi)
    stiffness_matrix = np.einsum("q,qia,qjb->abij", wts, curl, curl)

    # 3x3 Tensor is symmetric 
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    out = []
    for src in (mass_matrix, stiffness_matrix):
        for a, b in SYM:
            out.append(src[a, b] if a == b else src[a, b] + src[b, a])
    return np.ascontiguousarray(np.array(out))


def permutation_classes(mesh):
    """(class index per cell, unique permutation codes). uniq gives the permutation class codes for each cell, and cls gives the index of the class for each cell.
    For cell(c) clas[c] = k means that cell c belongs to the k-th permutation class, and uniq[k] gives the unique permutation code for that class."""
    mesh.topology.create_entity_permutations()
    perm = mesh.topology.get_cell_permutation_info()
    uniq, cls = np.unique(perm, return_inverse=True)
    return cls.astype(np.int32), uniq.astype(np.uint32)



def dense_transform(V, which, uperms):
    """Function returns the transformation matrices for each cell.
    A_cell = T · (reference matrix) · Tᵗ. Transformation Matrix is given via basix."""

    dof_dim = V.dofmap.list.shape[1] # Number of dofs per cell
    out = np.empty((len(uperms), dof_dim, dof_dim))
    element = V.element

    if which == "I" or not element.needs_dof_transformations:
        out[:] = np.eye(dof_dim)
        return out

    fn = {
        "T": element.T_apply,
        "Tt": element.Tt_apply,
        "Tt_inv": element.Tt_inv_apply,
    }[which]

    for k, p in enumerate(uperms):
        buf = np.ascontiguousarray(np.eye(dof_dim))
        fn(buf.reshape(-1), np.array([p], dtype=np.uint32), dof_dim)
        out[k] = buf
    return out

def piola_map(mesh):
    """Compute the coefficients for each cell, which are used to transform the reference matrix to the actual cell matrix. Jacobian is constant as we are using affine tetrahedra.
    12 coefficients per cell: 6 for the curl-curl part, 6 for the mass part."""
    v = mesh.geometry.x[mesh.geometry.dofmaps[0]] # Physical coords, shape is (n_cells, nodes per cell, dim)
    e0 = v[:, 1] - v[:, 0]
    e1 = v[:, 2] - v[:, 0]
    e2 = v[:, 3] - v[:, 0]

    J = np.stack([e0, e1, e2], axis=-1)
    detJ = np.abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)

    G = np.einsum("cab,cdb->cad", Jinv, Jinv) # G = inv(J) times inv(J) transposed
    H = np.einsum("cba,cbd->cad", J, J) # H = J transposed times J

    coef = np.empty((len(J), 12))
    SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for k, (a, b) in enumerate(SYM):
        coef[:, k]     = detJ * G[:, a, b]
        coef[:, k + 6] = H[:, a, b] / detJ
    return coef

degree = 3
quadrature_degree = 2 * degree + 2

element = basix.create_element(
    ElementFamily.N1E,
    CellType.tetrahedron,
    degree,
    lagrange_variant=LagrangeVariant.legendre
)
ufl_element = basix.ufl.wrap_element(element)

n = 4
cube_mesh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
V = fem.functionspace(cube_mesh, ufl_element)
n_cells = cube_mesh.topology.index_map(3).size_local

clas, uperms = permutation_classes(cube_mesh) # These are unique codes that rank the pattern. For tets there are max 4! ways to permute the verticies. Class 
refs = curlcurl_reference_matrix(element, quadrature_degree)

T  = dense_transform(V, "T",  uperms)
Tt = dense_transform(V, "Tt", uperms)

cell_coef = piola_map(cube_mesh)

# Folded generates a list of possible transformation matricies for each cell. The actual cell matrices are then given by A_cell[c] = folded[clas[c]] * cell_coef[c].
 
folded_ref = np.einsum("pai,kij,pjb->pkab", T, refs, Tt, optimize=True) # T R T^T

dofmap = V.dofmap.list
ndofs = V.dofmap.index_map.size_local
dofs_per_cell = dofmap.shape[1]


def matvec(x):
    """y = A x, without ever forming A (or even A_cell)."""
    x_local = x[dofmap]
    y_local = np.einsum("ck,ckab,cb->ca", cell_coef, folded_ref[clas], x_local)
    return np.bincount(dofmap.ravel(), weights=y_local.ravel(), minlength=ndofs)


# Check against the assembled operator.
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = form((ufl.inner(ufl.curl(u), ufl.curl(v)) + ufl.inner(u, v)) * ufl.dx)
A = assemble_matrix(a)
A.scatter_reverse()

x = np.random.default_rng(0).standard_normal(ndofs)
reference = A.to_scipy() @ x
mine = matvec(x)
print(f"dofs {ndofs}, cells {n_cells}, classes {len(uperms)}")
print("relative error:", np.linalg.norm(mine - reference) / np.linalg.norm(reference))


# Interpolation operator - Matrix Free

e_low = basix.create_element(
    ElementFamily.N1E, CellType.tetrahedron, 1, lagrange_variant=LagrangeVariant.legendre
)
e_high = basix.create_element(
    ElementFamily.N1E, CellType.tetrahedron, 2, lagrange_variant=LagrangeVariant.legendre
)

ufl_element_low = basix.ufl.wrap_element(e_low)
ufl_element_high = basix.ufl.wrap_element(e_high)

V_low = fem.functionspace(cube_mesh, ufl_element_low)
V_high = fem.functionspace(cube_mesh, ufl_element_high)

ndofs_low = V_low.dofmap.index_map.size_local
ndofs_high = V_high.dofmap.index_map.size_local

interp_matrix = interpolation_matrix(V_low, V_high)
interp_matrix.assemble()
print(f"interpolation matrix size: {interp_matrix.getSize()}")


# Get global DoF indices for cell 0
dofs_low_cell0 = V_low.dofmap.cell_dofs(0)
dofs_high_cell0 = V_high.dofmap.cell_dofs(0)

print(f"Cell 0 low-order global DoF indices: {dofs_low_cell0}")
print(f"Cell 0 high-order global DoF indices: {dofs_high_cell0}")


def fold_interpolation(V_src, V_dst, element_src, element_dst, uperms):
    """
        B_folded[p] = Td_p · B_ref · Ts_p. 
    """

    B_ref_cell = np.ascontiguousarray(basix.compute_interpolation_operator(element_src, element_dst))
    Ts = dense_transform(V_src, "Tt", uperms) 
    Td = dense_transform(V_dst, "Tt_inv", uperms)

    return np.einsum("pdi,ij,pjs->pds", Td, B_ref_cell, Ts, optimize=True)

B_folded = fold_interpolation(V_low, V_high, e_low, e_high, uperms)


def interpolation_mat_free(x, V_low, V_high):
    dofmap_low = V_low.dofmap.list
    dofmap_high = V_high.dofmap.list
    mult = np.bincount(dofmap_high.ravel(), minlength=ndofs_high)[:ndofs_high].astype(float)

    y_cell = np.einsum("cij,cj->ci", B_folded[clas], x[dofmap_low], optimize=True)
    y = np.bincount(dofmap_high.ravel(), weights=y_cell.ravel(), minlength=ndofs_high)
    return y[:ndofs_high] / mult


rng = np.random.default_rng(0)
x = rng.standard_normal(ndofs_low)
y_mf = interpolation_mat_free(x, V_low, V_high)

x_petsc = interp_matrix.createVecRight()   # size = ndofs_low
y_petsc = interp_matrix.createVecLeft()    # size = ndofs_high
x_petsc.array_w[:] = x
interp_matrix.mult(x_petsc, y_petsc)
y_assembled = y_petsc.array_r.copy()

# Compare
error = np.linalg.norm(y_mf - y_assembled)
relative_error = error / np.linalg.norm(y_assembled)

print("||B_mf x - B x|| =", error)
print("relative error   =", relative_error)

def discrete_gradient_reference_matrix(element_h1, element_curl):

    pts = element_curl.points # Nédélec interpolation points
    nq, tdim = pts.shape
    tab = element_h1.tabulate(1, pts) 
    dphi = tab[1:, :, :, 0].reshape(tdim * nq, -1)   # (tdim*nq, n0)
    ref_mat = element_curl.interpolation_matrix @ dphi
    return ref_mat

# Test 

element_h1 = basix.create_element(
    ElementFamily.P, CellType.tetrahedron, degree,
    lagrange_variant=LagrangeVariant.gll_warped
)

element_curl = basix.create_element(
    ElementFamily.N1E, CellType.tetrahedron, degree,
    lagrange_variant=LagrangeVariant.legendre
)

discrete_grad_ref = discrete_gradient_reference_matrix(element_h1, element_curl)



def fold_discrete_gradient(V_lagrange, V_nedelec, element_lagrange, element_nedelec, uperms):
    # Step 1: reference-cell matrix, shape (140, 56)
    G_ref = discrete_gradient_reference_matrix(element_lagrange, element_nedelec)

    # Step 2: transformation matrices, one per class
    Ts = dense_transform(V_lagrange, "Tt", uperms)      # shape (n_classes, 56, 56)
    Td = dense_transform(V_nedelec, "Tt_inv", uperms)   # shape (n_classes, 140, 140)

    # Step 3: for each class, sandwich G_ref between its two transformations
    n_classes = len(uperms)
    G_folded = np.zeros((n_classes, G_ref.shape[0], G_ref.shape[1]))   # (n_classes, 140, 56)

    for p in range(n_classes):
        G_folded[p] = Td[p] @ G_ref @ Ts[p]    # (140,140) @ (140,56) @ (56,56) → (140,56)

    return G_folded


V_lag = fem.functionspace(cube_mesh, basix.ufl.wrap_element(element_h1))
V_ned = fem.functionspace(cube_mesh, basix.ufl.wrap_element(element_curl))

dofmap_lag = V_lag.dofmap.list
dofmap_ned = V_ned.dofmap.list

n_dofs_ned = V_ned.dofmap.index_map.size_local


"""y = G x, one cell at a time."""
dofmap_lag = V_lag.dofmap.list
dofmap_ned = V_ned.dofmap.list
n_dofs_ned = V_ned.dofmap.index_map.size_local

im_lag = V_lag.dofmap.index_map
im_ned = V_ned.dofmap.index_map
n_lag_all = im_lag.size_local + im_lag.num_ghosts
n_ned_all = im_ned.size_local + im_ned.num_ghosts

x = np.random.default_rng(0).standard_normal(n_lag_all)   # an H1 vector, not the N1E one above
y = np.zeros(n_ned_all)                   # output vector, one entry per Nédélec DOF
G_folded = fold_discrete_gradient(V_lag, V_ned, element_h1, element_curl, uperms)

for c in range(n_cells):
    lag_dofs = dofmap_lag[c]
    ned_dofs = dofmap_ned[c]

    x_local = x[lag_dofs]

    k = clas[c]
    G_local = G_folded[k]

    y_local = G_local @ x_local  

    y[ned_dofs] = y_local

G_petsc = discrete_gradient(V_lag, V_ned)
G_petsc.assemble()


# yp (ned) = G * xp(Lag)
xp = G_petsc.createVecRight()
yp = G_petsc.createVecLeft()

n_lag = xp.getLocalSize()
xp.array_w[:] = x[:n_lag]

G_petsc.mult(xp, yp)                   # yp = G x
y_petsc = yp.array_r.copy()

print("Error is ", norm(y - y_petsc))

def boundary_dof_mask(V):
    """Gives an array 0s and 1s"""
    msh = V.mesh
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)

    mask = np.zeros(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts)
    #Identify all the dofs on the boundary
    mask[bdofs] = 1
    return mask


bc_mask = boundary_dof_mask(V)[:ndofs]      # M:    1 on the boundary, 0 inside
keep_mask = 1.0 - bc_mask                   # I - M: 0 on the boundary, 1 inside
print(f"bc dofs: {int(bc_mask.sum())} of {ndofs}")


def cell_matrix(c):
    """ Cell wise matrix"""
    k_mats = folded_ref[clas[c]]
    A_cell = np.zeros((dofs_per_cell, dofs_per_cell))
    for k in range(k_mats.shape[0]):
        A_cell += cell_coef[c, k] * k_mats[k]
    return A_cell

#%%

def matvec_bc(x):
    """ y = ax with dirichlet Bcs"""
    x_in = keep_mask * x      

    y = np.zeros(ndofs)
    for c in range(n_cells):
        dofs = dofmap[c]
        y[dofs] += cell_matrix(c) @ x_in[dofs] 

    return keep_mask * y + bc_mask * x  


def diagonal_bc():
    d = np.zeros(ndofs)
    for c in range(n_cells):
        dofs = dofmap[c]
        d[dofs] += np.diag(cell_matrix(c))

    return keep_mask * d + bc_mask      # 1 on the bc diagonal


g = fem.Function(V)
bc = fem.dirichletbc(g, np.flatnonzero(boundary_dof_mask(V)))
A_bc = assemble_matrix(a, bcs=[bc])
A_bc.scatter_reverse()

x = np.random.default_rng(0).standard_normal(ndofs)
A_bc_sparse = A_bc.to_scipy()
ref = A_bc_sparse @ x
print("matvec_bc relative error:",
      np.linalg.norm(matvec_bc(x) - ref) / np.linalg.norm(ref))

ref_diag = A_bc_sparse.diagonal()
print("diagonal_bc relative error:",
      np.linalg.norm(diagonal_bc() - ref_diag) / np.linalg.norm(ref_diag))
