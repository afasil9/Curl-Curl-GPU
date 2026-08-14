"""Poisson convergence study on the GPU (PETSc aijcusparse + hypre BoomerAMG).
    Solving:

    -div(grad u) =: f,   

For Lagrange degree p the expected L2 rate is p+1 and the H1 rate is p.
To run:

PETSC_OPTIONS="-use_gpu_aware_mpi 0" python3 poisson.py [-log_view ...]
"""

import sys

import petsc4py

petsc4py.init(sys.argv)

import numpy as np
import ufl
from dolfinx.common import Timer
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.mesh import create_unit_cube, exterior_facet_indices
from mpi4py import MPI
from petsc4py import PETSc
from utils import L2_norm, monitor

degree = 1
n = 40
mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n)

V = functionspace(mesh, ("Lagrange", degree))
tdim = mesh.topology.dim

mesh.topology.create_connectivity(tdim - 1, tdim)
facets = exterior_facet_indices(mesh.topology)
dofs = locate_dofs_topological(V=V, entity_dim=tdim - 1, entities=facets)
bc = dirichletbc(value=Constant(mesh, 0.0), dofs=dofs, V=V)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)

u_ex = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2])
f = -ufl.div(ufl.grad(u_ex))

a = form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
L = form(ufl.inner(f, v) * ufl.dx)

# Device matrix: assembled on host, mirrored to the GPU by PETSc
t = Timer("Assemble matrix (aijcusparse)")
A = assemble_matrix(a, bcs=[bc], kind="aijcusparse")
A.assemble()
del t

b = A.createVecRight()
sol_vector = A.createVecLeft()
uh = Function(V)

t = Timer("Assemble vector")
with b.localForm() as b_loc:
    b_loc.set(0.0)
assemble_vector(b, L)
apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])
del t

ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.CG)

ksp.setTolerances(rtol=1e-12)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.HYPRE)
pc.setHYPREType("boomeramg")
ksp.setFromOptions()

t = Timer("KSP setup")
ksp.setUp()
del t

t = Timer("Solve (CG + hypre BoomerAMG on GPU)")
# ksp.setMonitor(monitor)
ksp.solve(b, sol_vector)
del t

sol_vector.copy(uh.x.petsc_vec)
uh.x.scatter_forward()

print(f"ksp reason: {ksp.getConvergedReason()}")
print(f"ksp iterations: {ksp.getIterationNumber()}")
error = uh - u_ex
print(f"L2 norm is {L2_norm(error):.8e}")
