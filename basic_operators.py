#%%
import basix
import numpy as np
from basix import CellType, ElementFamily, LagrangeVariant
from dolfinx import mesh, fem
from mpi4py import MPI
import ufl
from dolfinx.fem import form, assemble_matrix

def curlcurl_reference_matrix(element, quadrature_degree=2):

    """Computes reference matrix over a reference tet"""
    pts, wts = basix.make_quadrature(element.cell_type, quadrature_degree)

    tab = element.tabulate(1, pts) #Evaluate the element basis functions and their first derivatives at the quadrature points.
    phi = tab[0]
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
    """A_cell = T · (reference matrix) · Tᵗ. Transformation Matrix is given via basix."""

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
    v = mesh.geometry.x[mesh.geometry.dofmaps[0]]
    e0 = v[:, 1] - v[:, 0]
    e1 = v[:, 2] - v[:, 0]
    e2 = v[:, 3] - v[:, 0]

    J = np.stack([e0, e1, e2], axis=-1)
    detJ = np.abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)

    G = np.einsum("cab,cdb->cad", Jinv, Jinv)
    H = np.einsum("cba,cbd->cad", J, J)

    coef = np.empty((len(J), 12))
    for k, (a, b) in enumerate(SYM):
        coef[:, k]     = detJ * G[:, a, b]
        coef[:, k + 6] = H[:, a, b] / detJ
    return coef

degree = 4
quadrature_degree = 2 * degree + 2
element = basix.create_element(
    ElementFamily.N1E,
    CellType.tetrahedron,
    degree,
    lagrange_variant=LagrangeVariant.legendre
)


SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]   # same order as your reference matrices

n = 10
cube_mesh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
V = fem.functionspace(cube_mesh, ("N1E", degree))
number_of_cells = cube_mesh.topology.index_map(3).size_local

clas, uperms = permutation_classes(cube_mesh) # These are unique codes that rank the pattern. For tets there are max 4! ways to permute the verticies. Class 
refs = curlcurl_reference_matrix(element, quadrature_degree)

T  = dense_transform(V, "T",  uperms)
Tt = dense_transform(V, "Tt", uperms)

cell_coef = piola_map(cube_mesh)
folded = np.einsum("pai,kij,pjb->pkab", T, refs, Tt, optimize=True) # T R T^T

# Folded will give the cell matrices for each permutation class. The actual cell matrices are then given by A_cell[c] = folded[clas[c]] * cell_coef[c]

dofmap = V.dofmap.list
ndofs = V.dofmap.index_map.size_local
dofs_per_cell = dofmap.shape[1]

# This will give the cell ordering 
cube_mesh.geometry.dofmaps[0] = dofmap  

def matvec(x):
    """y = A x, without ever forming A (or even A_cell)."""
    x_local = x[dofmap]
    y_local = np.einsum("ck,ckab,cb->ca", cell_coef, folded[clas], x_local)
    return np.bincount(dofmap.ravel(), weights=y_local.ravel(), minlength=ndofs)


# Check against the assembled operator. Keep n small: to_dense() is O(ndofs^2).
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = form((ufl.inner(ufl.curl(u), ufl.curl(v)) + ufl.inner(u, v)) * ufl.dx)
A = assemble_matrix(a)
A.scatter_reverse()

x = np.random.default_rng(0).standard_normal(ndofs)
reference = A.to_dense() @ x
mine = matvec(x)
print(f"dofs {ndofs}, cells {number_of_cells}, classes {len(uperms)}")
print("relative error:", np.linalg.norm(mine - reference) / np.linalg.norm(reference))

