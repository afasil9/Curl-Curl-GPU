import ctypes
import sys
import numpy as np
from dolfinx.fem import (
    assemble_scalar,
    form,
    Expression,
    Function,
    functionspace,
)
from mpi4py import MPI
from ufl import dx, inner
from ufl.core.expr import Expr
import ufl


def hypre_use_vendor_spgemm(use_vendor, libname="libHYPRE-3.1.0.so"):
    """Set to use which vendor for Sparse general matrix-matrix multiplication. Set to 0 to use hypre's SpGEMM, set to 1 to use cuSPARSE's SpGEMM. The vendor one runs out of resources on the AMS setup products for higher degrees."""
    lib = ctypes.CDLL(libname)
    if not lib.HYPRE_Initialized():
        lib.HYPRE_Initialize()
    lib.HYPRE_SetSpGemmUseVendor(ctypes.c_int(int(use_vendor)))

JIT_OPTIONS = {
    "cffi_extra_compile_args": [
        "-O3",
        "-march=native",
        "-fno-math-errno",
        "-fassociative-math",
        "-fno-signed-zeros",
        "-fno-trapping-math",
        "-g0",
    ],
    "timeout": 60,
}

def run_header(comm, backend, degree, n, V):
    """Print a machine-parsable description of the run at the top of the log."""
    mesh = V.mesh
    ndofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    ncells = mesh.topology.index_map(mesh.topology.dim).size_global
    par_print(comm, "=== run parameters ===")
    par_print(comm, f"backend: {backend}")
    par_print(comm, f"degree: {degree}")
    par_print(comm, f"n: {n}")
    par_print(comm, f"ncells: {ncells}")
    par_print(comm, f"ndofs: {ndofs}")
    par_print(comm, f"ranks: {comm.size}")
    par_print(comm, "======================")

def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(
        MPI.COMM_WORLD.allreduce(assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM)
    )

def monitor(ksp, its, rnorm):
    """KSP monitor printing the preconditioned residual on rank 0 only"""
    par_print(ksp.comm.tompi4py(), f"Iteration: {its}, preconditioned residual: {rnorm}")



def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0)) for i in range(3)
    ]
    return np.logical_or(np.logical_or(boundaries[0], boundaries[1]), boundaries[2])


def error_L2(uh, u_ex, degree_raise=4):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(inner(e_W, e_W) * dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

