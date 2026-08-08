"""
Curl-curl + mass solved on the GPU with PETSc, p-multigrid preconditioned CG.
curl(curl(u)) + u = f  on the unit cube,  u x n given on the boundary. Hiptmair jacobi preconditioner at the highest levels with a Chebyshev smoother, solving residual correction equation
The error At the lowest level, AMS preconditioner is used. 
The matricies for each order are assembled.
    
To run on GPU, use the following command:
    PETSC_OPTIONS="-use_gpu_aware_mpi 0" python p_multigrid.py
"""

import petsc4py
import sys

petsc4py.init(sys.argv)

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import dolfinx
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from ufl import (
    curl,
    grad,
    inner,
    SpatialCoordinate,
    TrialFunction,
    TestFunction,
    dx,
    as_vector,
    sin,
    pi,
)
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    create_vector,
    set_bc,
)
from dolfinx.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.mesh import exterior_facet_indices
from utils import L2_norm, hypre_use_vendor_spgemm, par_print, run_header, JIT_OPTIONS

PETSc.Log.begin()


class HiptmairJacobi:
    """Point Jacobi in the Nedelec space + point Jacobi in the gradient space.
    Jacobi preconditioner is simply just the inverse diagonal.
    The gradient space is inside the Nedelec space exactly, the correction is given as:
    y <- Diag r + G diag(K)^-1 G^T r
    """

    def __init__(self, G, aux_dinv):
        self.G = G
        self.aux_dinv = aux_dinv
        self._r_aux = G.createVecRight()
        self._z_aux = G.createVecRight()
        self._y_aux = G.createVecLeft()

    def setUp(self, pc):
        A, _ = pc.getOperators()
        self.dinv = A.createVecRight()
        A.getDiagonal(self.dinv)
        self.dinv.reciprocal()

    def apply(self, pc, r, y):
        y.pointwiseMult(self.dinv, r) # Point Jacobi in Nedelec space
        self.G.multTranspose(r, self._r_aux)  # _r_aux = G^T r
        self._z_aux.pointwiseMult(self.aux_dinv, self._r_aux)  # diag(K)^-1 G^T r
        self.G.mult(self._z_aux, self._y_aux)
        y.axpy(1.0, self._y_aux)  # add the gradient-space correction


n = 8
degree = 4
degrees = list(range(1, degree + 1))  # unit-step p-ladder: 1, 2, ..., degree
mat_type = "aijcusparse"  # "aijcusparse" for GPU or "aij" for CPU

smoother = "hiptmair"  # "hiptmair" or "jacobi"
smoother_its = 3
esteig = "0,0.1,0,1.1"
rtol = 1e-10
max_it = 1000
monitor = False

comm = MPI.COMM_WORLD
mesh = dolfinx.mesh.create_unit_cube(comm, n, n, n)
tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
facets = exterior_facet_indices(mesh.topology)

x = SpatialCoordinate(mesh)
u_ex = as_vector(
    (
        sin(pi * x[1]) * sin(pi * x[2]),
        sin(pi * x[2]) * sin(pi * x[0]),
        sin(pi * x[0]) * sin(pi * x[1]),
    )
)
f = curl(curl(u_ex)) + u_ex

Vs = []  # function spaces, coarse -> fine
As = []  # operators, one per level
forms = []  # bilinear forms, one per level
bcs = []  # Dirichlet bc, one per level
bc_dofs = []  # constrained dof indices, one per level

for d in degrees:
    V = functionspace(mesh, ("N1curl", d))
    dofs = locate_dofs_topological(V, tdim - 1, facets)

    u_bc = Function(V)
    u_bc.interpolate(Expression(u_ex, V.element.interpolation_points))
    bc = dirichletbc(u_bc, dofs)

    u, v = TrialFunction(V), TestFunction(V)
    a = form(inner(curl(u), curl(v)) * dx + inner(u, v) * dx, jit_options=JIT_OPTIONS)

    t = dolfinx.common.Timer(f"Assemble matrix degree {d}")
    A = assemble_matrix(a, bcs=[bc], kind=mat_type)
    A.assemble()
    del t

    Vs.append(V)
    As.append(A)
    forms.append(a)
    bcs.append(bc)
    bc_dofs.append(dofs)

    ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    par_print(comm, f"level degree {d}: {ndofs} dofs")

nlevels = len(degrees)
V_fine = Vs[-1]
A_fine = As[-1]
a_fine = forms[-1]
bc_fine = bcs[-1]

masks = [] # masks for each level, where 1 = free dof, 0 = constrained dof
for V, dofs in zip(Vs, bc_dofs):
    bs = V.dofmap.index_map_bs
    n_owned = V.dofmap.index_map.size_local * bs
    blocked = (dofs[:, None] * bs + np.arange(bs)).ravel()
    mask = np.ones(n_owned, dtype=PETSc.ScalarType)
    mask[blocked[blocked < n_owned]] = 0.0
    masks.append(mask)

prolongations = []
for i in range(1, nlevels):
    # y = P x  (x in coarse space, y in fine space)
    P = interpolation_matrix(Vs[i - 1], Vs[i])
    P.assemble()

    # P is not constrained, so we need to zero out the rows and columns

    left, right = P.createVecLeft(), P.createVecRight()
    left.array[:] = masks[i]
    right.array[:] = masks[i - 1]

    # Build prolongation with zero rows/columns for constrained dofs
    P.diagonalScale(L=left, R=right)

    if mat_type != P.getType():
        P.convert(mat_type, P)
    prolongations.append(P)

run_header(comm, "pmg", degree, n, V_fine)

L = form(inner(f, TestFunction(V_fine)) * dx, jit_options=JIT_OPTIONS)

if mat_type == "aijcusparse":
    b = A_fine.createVecRight()
    b.setType(PETSc.Vec.Type.CUDA)
else:
    b = create_vector(V_fine)
    b.set(0.0)

t = dolfinx.common.Timer("Assemble vector")
assemble_vector(b, L)
apply_lifting(b, [a_fine], bcs=[[bc_fine]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc_fine])
del t

xv = A_fine.createVecLeft()
if mat_type == "aijcusparse":
    xv.setType(PETSc.Vec.Type.CUDA)

ksp = PETSc.KSP().create(comm)
ksp.setOptionsPrefix("pmg_")
ksp.setOperators(A_fine)
ksp.setType(PETSc.KSP.Type.CG)
ksp.setTolerances(rtol=rtol, max_it=max_it)
ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)

pc = ksp.getPC()
pc.setType("mg")
pc.setMGLevels(nlevels)
pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
pc.setMGCycleType(PETSc.PC.MGCycleType.V)

for i, P in enumerate(prolongations, start=1):
    pc.setMGInterpolation(i, P)

for i in range(nlevels):
    pc.getMGSmoother(i).setOperators(As[i])

# Chebyshev smoothers on every level but the coarsest
opts = PETSc.Options()
opts.prefixPush(ksp.getOptionsPrefix())
opts["mg_levels_ksp_type"] = "chebyshev"
opts["mg_levels_ksp_max_it"] = smoother_its
opts["mg_levels_ksp_chebyshev_esteig"] = esteig
opts["mg_levels_esteig_ksp_type"] = "gmres"
opts["mg_levels_esteig_ksp_max_it"] = 20
if smoother == "jacobi" and nlevels > 1:
    opts["mg_levels_pc_type"] = "jacobi"
if monitor:
    opts["ksp_monitor_true_residual"] = None
opts.prefixPop()

coarse_ksp = pc.getMGCoarseSolve()
coarse_ksp.setType(PETSc.KSP.Type.PREONLY)
coarse_pc = coarse_ksp.getPC()
coarse_pc.setType("hypre")
coarse_pc.setHYPREType("ams")

if mat_type == "aijcusparse":
    hypre_use_vendor_spgemm(0)

V_CG_coarse = functionspace(mesh, ("CG", degrees[0]))
G_ams = discrete_gradient(V_CG_coarse, Vs[0])
G_ams.assemble()
if mat_type != G_ams.getType():
    G_ams.convert(mat_type, G_ams)
coarse_pc.setHYPREDiscreteGradient(G_ams)

Vec_CG_coarse = functionspace(mesh, ("CG", degrees[0], (tdim,)))
Pi_ams = interpolation_matrix(Vec_CG_coarse, Vs[0])
Pi_ams.assemble()
if mat_type != Pi_ams.getType():
    Pi_ams.convert(mat_type, Pi_ams)
coarse_pc.setHYPRESetInterpolations(tdim, ND_Pi_Full=Pi_ams)

opts[f"{coarse_ksp.getOptionsPrefix()}pc_hypre_ams_cycle_type"] = 1

ksp.setFromOptions()

smoother_refs = []

if smoother == "hiptmair":
    for i in range(1, nlevels):
        d = degrees[i]
        V = Vs[i]

        S = functionspace(mesh, ("CG", d))
        s_dofs = locate_dofs_topological(S, tdim - 1, facets)

        bs = S.dofmap.index_map_bs
        n_owned = S.dofmap.index_map.size_local * bs
        blocked = (s_dofs[:, None] * bs + np.arange(bs)).ravel()
        s_mask = np.ones(n_owned, dtype=PETSc.ScalarType)
        s_mask[blocked[blocked < n_owned]] = 0.0

        G = discrete_gradient(S, V)
        G.assemble()
        left, right = G.createVecLeft(), G.createVecRight()
        left.array[:] = masks[i]
        right.array[:] = s_mask
        G.diagonalScale(L=left, R=right)
        if mat_type != G.getType():
            G.convert(mat_type, G)

        # Diagonal of the CG stiffness matrix -> Jacobi in the gradient space
        p, q = TrialFunction(S), TestFunction(S)
        s_bc = dirichletbc(Constant(mesh, PETSc.ScalarType(0.0)), s_dofs, S)
        K = assemble_matrix(
            form(inner(grad(p), grad(q)) * dx, jit_options=JIT_OPTIONS),
            bcs=[s_bc],
            kind=mat_type,
        )
        K.assemble()
        aux_dinv = K.createVecRight()
        K.getDiagonal(aux_dinv)
        aux_dinv.reciprocal()
        K.destroy()

        level_pc = pc.getMGSmoother(i).getPC()
        level_pc.setType(PETSc.PC.Type.PYTHON)
        level_pc.setPythonContext(HiptmairJacobi(G, aux_dinv))
        smoother_refs.append((G, aux_dinv))

t = dolfinx.common.Timer("KSP setup")
ksp.setUp()
del t

t = dolfinx.common.Timer("Solve (CG + p-multigrid on GPU)")
ksp.solve(b, xv)
del t

reason = ksp.getConvergedReason()
if reason < 0:
    raise RuntimeError(f"KSP failed to converge, reason {reason}")

uh = Function(V_fine)
xv.copy(uh.x.petsc_vec)
uh.x.scatter_forward()


dolfinx.common.list_timings(comm)
PETSc.Log.view()

par_print(comm, f"ksp reason: {reason}")
par_print(comm, f"ksp iterations: {ksp.getIterationNumber()}")
par_print(comm, f"L2 norm is {L2_norm(curl(uh - u_ex)):.8e}")
