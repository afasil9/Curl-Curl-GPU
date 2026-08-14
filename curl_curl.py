
"""Curl-curl + mass solved on the GPU with PETSc + hypre.
To run:
PETSC_OPTIONS="-use_gpu_aware_mpi 0" python curl_curl.py 

To run with GPU timings:
PETSC_OPTIONS="-use_gpu_aware_mpi 0 -log_view_gpu_time" python curl_curl.py > output_gpu.txt 2>&1

"""


from mpi4py import MPI
from petsc4py import PETSc
import petsc4py
import sys
petsc4py.init(sys.argv)

import numpy as np
import ufl

import dolfinx
from dolfinx.fem import (
    Expression,
    Function,
    form,
    functionspace,
    locate_dofs_topological,
)
from ufl import curl, inner, SpatialCoordinate, TrialFunction, TestFunction, dx, as_vector, sin, SpatialCoordinate, pi
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.mesh import exterior_facet_indices
from utils import JIT_OPTIONS, L2_norm, hypre_use_vendor_spgemm
from dolfinx.fem.petsc import discrete_gradient, interpolation_matrix

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

# Device matrix: assembled on host, mirrored to the GPU by PETSc
t = dolfinx.common.Timer("Assemble matrix (aijcusparse)")
A = assemble_matrix(a, bcs=[bc], kind="aijcusparse")
A.assemble()
del t

t = dolfinx.common.Timer("Assemble vector")
b_host = assemble_vector(L)
apply_lifting(b_host, [a], bcs=[[bc]])
b_host.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b_host, [bc])
del t

b = A.createVecRight()
b.setType(PETSc.Vec.Type.CUDA)
b.setArray(b_host.getArray(readonly=True))

xv = A.createVecLeft()
xv.set(0.0)

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
pc.setType("hypre")
pc.setHYPREType("ams")

hypre_use_vendor_spgemm(0) # Use Hypres SPGemm. CuSparse runs into VRAM issues for large problems.

# AMS auxiliary operators. dolfinx builds these as host AIJ. We need to convert them into aijcusparse
V_CG = functionspace(mesh, ("CG", degree))
G = discrete_gradient(V_CG, V)
G.assemble()
G.convert(PETSc.Mat.Type.AIJCUSPARSE, G)
pc.setHYPREDiscreteGradient(G)

Vec_CG = functionspace(mesh, ("CG", degree, (tdim,)))
Pi = interpolation_matrix(Vec_CG, V)
Pi.assemble()
Pi.convert(PETSc.Mat.Type.AIJCUSPARSE, Pi)
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

t = dolfinx.common.Timer("Solve (CG + hypre on GPU)")
ksp.solve(b, xv)
del t

reason = ksp.getConvergedReason()
if reason < 0:
    raise RuntimeError(f"KSP failed to converge, reason {reason}")

uh = Function(V)

xv.copy(uh.x.petsc_vec)
uh.x.scatter_forward()
error = uh - u_ex

print(f"ksp reason: {ksp.getConvergedReason()}")
print(f"ksp iterations: {ksp.getIterationNumber()}")
print(f"L2 norm is {L2_norm(curl(error)):.8e}")

dolfinx.common.list_timings(comm)
PETSc.Log.view()