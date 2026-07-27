from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl

import dolfinx
from dolfinx.fem import (
    Expression,
    Function,
    assemble_scalar,
    form,
    functionspace,
    locate_dofs_topological,
)
from ufl import curl, inner, SpatialCoordinate, TrialFunction, TestFunction, dx, as_vector, sin, SpatialCoordinate, pi
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.mesh import exterior_facet_indices
from utils import L2_norm

degree = 1

n = 8

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
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

a = form(inner(curl(u), curl(v)) * dx + inner(u, v) * dx)
L = form(inner(f, v) * dx)

# Device matrix: assembled on host, mirrored to the GPU by PETSc
t = dolfinx.common.Timer("Assemble matrix (aijcusparse)")
A = assemble_matrix(a, bcs=[bc], kind="aijcusparse")
A.assemble()
del t

# Device vectors
b = A.createVecRight()
b.setType(PETSc.Vec.Type.CUDA)
b.set(0.0)
uh = Function(V, )

xv = A.createVecLeft()
xv.setType(PETSc.Vec.Type.CUDA)

t = dolfinx.common.Timer("Assemble vector")
assemble_vector(b, L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])
del t

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)
ksp.setTolerances(rtol=1e-10, max_it=10000)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")
ksp.setFromOptions()

t = dolfinx.common.Timer("KSP setup")
ksp.setUp()
del t

t = dolfinx.common.Timer("Solve (CG + hypre on GPU)")
ksp.solve(b, xv)
del t

reason = ksp.getConvergedReason()
if reason < 0:
    raise RuntimeError(f"KSP failed to converge, reason {reason}")

xv.copy(uh.x.petsc_vec)
uh.x.scatter_forward()
error = uh - u_ex

print(f"ksp reason: {ksp.getConvergedReason()}")
print(f"ksp iterations: {ksp.getIterationNumber()}")
print(f"L2 norm is {L2_norm(curl(error)):.8e}")

# dolfinx.common.list_timings(MPI.COMM_WORLD)
