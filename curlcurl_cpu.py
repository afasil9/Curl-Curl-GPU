
"""Curl-curl + mass solved on the CPU with PETSc + hypre.

To run:

    OMP_NUM_THREADS=1 mpirun -n 8 python curlcurl_cpu.py > output_cpu.txt 2>&1

"""

import sys

import petsc4py
from mpi4py import MPI
from petsc4py import PETSc

petsc4py.init(sys.argv)

import dolfinx
from dolfinx.fem import (
    Expression,
    Function,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    discrete_gradient,
    interpolation_matrix,
    set_bc,
)
from dolfinx.mesh import exterior_facet_indices
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    curl,
    dx,
    inner,
    pi,
    sin,
)

from utils import JIT_OPTIONS, L2_norm, par_print

PETSc.Log.begin()

comm = MPI.COMM_WORLD
degree = 2
n = 16

mesh = dolfinx.mesh.create_unit_cube(comm, n, n, n)
V = functionspace(mesh, ("N1curl", degree))
tdim = mesh.topology.dim

x = SpatialCoordinate(mesh)
u_ex = as_vector(
    (
        sin(pi * x[1]) * sin(pi * x[2]),
        sin(pi * x[2]) * sin(pi * x[0]),
        sin(pi * x[0]) * sin(pi * x[1]),
    )
)
f = curl(curl(u_ex)) + u_ex

mesh.topology.create_connectivity(tdim - 1, tdim)
facets = exterior_facet_indices(mesh.topology)
dofs = locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)

u_bc = Function(V)
u_bc.interpolate(Expression(u_ex, V.element.interpolation_points))
bc = dolfinx.fem.dirichletbc(u_bc, dofs)

u = TrialFunction(V)
v = TestFunction(V)

# a = form(inner(curl(u), curl(v)) * dx + inner(u, v) * dx)
# L = form(inner(f, v) * dx)
a = form(inner(curl(u), curl(v)) * dx + inner(u, v) * dx, jit_options=JIT_OPTIONS)
L = form(inner(f, v) * dx, jit_options=JIT_OPTIONS)

t = dolfinx.common.Timer("Assemble matrix")
A = assemble_matrix(a, bcs=[bc])
A.assemble()
del t

# Device vectors
t = dolfinx.common.Timer("Assemble vector")
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])
del t

uh = Function(V)

xv = A.createVecLeft()

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.setTolerances(rtol=1e-10, max_it=10000)
pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

V_CG = functionspace(mesh, ("CG", degree))
G = discrete_gradient(V_CG, V)
G.assemble()
pc.setHYPREDiscreteGradient(G)

Vec_CG = functionspace(mesh, ("CG", degree, (tdim,)))
Pi = interpolation_matrix(Vec_CG, V)
Pi.assemble()
pc.setHYPRESetInterpolations(tdim, ND_Pi_Full=Pi)

PETSc.Options()["pc_hypre_ams_cycle_type"] = 7
PETSc.Options()["pc_hypre_ams_tol"] = 1e-8
PETSc.Options()["pc_hypre_ams_max_iter"] = 1
PETSc.Options()["pc_hypre_ams_amg_beta_theta"] = 0.25
PETSc.Options()["pc_hypre_ams_print_level"] = 1

opts = PETSc.Options()
prefix = ksp.getOptionsPrefix() or ""
# opts[f"{prefix}ksp_monitor_true_residual"] = None

ksp.setFromOptions()

t = dolfinx.common.Timer("KSP setup")
ksp.setUp()
del t

t = dolfinx.common.Timer("Solve (CG + hypre on CPU)")
ksp.solve(b, xv)
del t

reason = ksp.getConvergedReason()
if reason < 0:
    raise RuntimeError(f"KSP failed to converge, reason {reason}")

xv.copy(uh.x.petsc_vec)
uh.x.scatter_forward()
error = uh - u_ex

dolfinx.common.list_timings(comm)
PETSc.Log.view()

par_print(comm, f"ksp reason: {ksp.getConvergedReason()}")
par_print(comm, f"ksp iterations: {ksp.getIterationNumber()}")
par_print(comm, f"L2 norm is {L2_norm(curl(error)):.8e}")