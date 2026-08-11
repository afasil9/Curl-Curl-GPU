"""Sweep the p-multigrid solver over a range of mesh sizes for a fixed degree ladder.

To run on GPU, use:
    PETSC_OPTIONS="-use_gpu_aware_mpi 0" python sweep.py
"""

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import Function
from ufl import curl

from p_multigrid import build_hierarchy, build_rhs, setup_pmg
from utils import L2_norm, par_print
import dolfinx
from dolfinx.fem import functionspace
from dolfinx.mesh import exterior_facet_indices
from ufl import SpatialCoordinate, as_vector, sin, pi
import numpy as np

import os
import sys
from contextlib import contextmanager
from pathlib import Path


CASES = {
    1: [1, 2, 3],
    2: [1, 3],
}


@contextmanager
def redirect_stdout_to(path, comm):
    sys.stdout.flush()
    target = path if comm.rank == 0 else os.devnull
    saved = os.dup(1)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.dup2(fd, 1)
        os.close(fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def solve(comm, n, degrees, mat_type, smoother, smoother_its, rtol, max_it, monitor):
    mesh = dolfinx.mesh.create_unit_cube(comm, n, n, n)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = exterior_facet_indices(mesh.topology)

    DGO_space = functionspace(mesh, ("DG", 0))
    alpha = Function(DGO_space)
    beta = Function(DGO_space)

    alpha.interpolate(lambda x: np.where(x[0] <= 0.5, 1.0, 1.0))
    beta.interpolate(lambda x: np.where(x[0] <= 0.5, 1.0, 1.0))

    x = SpatialCoordinate(mesh)
    u_ex = as_vector(
        (
            sin(pi * x[1]) * sin(pi * x[2]),
            sin(pi * x[2]) * sin(pi * x[0]),
            sin(pi * x[0]) * sin(pi * x[1]),
        )
    )
    f = curl(alpha * curl(u_ex)) + beta * u_ex

    Vs, As, forms, bcs, bc_dofs, prolongations, masks = build_hierarchy(
        mesh, degrees, alpha, beta, u_ex, facets, mat_type
    )

    V_fine, A_fine, a_fine, bc_fine = Vs[-1], As[-1], forms[-1], bcs[-1]

    b, xv = build_rhs(V_fine, A_fine, f, a_fine, bc_fine, mat_type)

    ksp = setup_pmg(
        mesh,
        degrees,
        Vs,
        As,
        prolongations,
        masks,
        facets,
        beta,
        mat_type,
        PETSc.KSP.Type.CG,
        smoother,
        smoother_its,
        rtol,
        max_it,
        monitor,
    )

    ksp.solve(b, xv)

    reason = ksp.getConvergedReason()
    if reason < 0:
        raise RuntimeError(f"KSP failed to converge at n={n}, reason {reason}")

    uh = Function(V_fine)
    xv.copy(uh.x.petsc_vec)
    uh.x.scatter_forward()

    result = {
        "n": n,
        "ndofs": V_fine.dofmap.index_map.size_global * V_fine.dofmap.index_map_bs,
        "its": ksp.getIterationNumber(),
        "reason": reason,
        "curl_l2": L2_norm(curl(uh - u_ex)),
    }

    # Release the PETSc objects before the next mesh size is built
    ksp.destroy()
    for A in As:
        A.destroy()
    for P in prolongations:
        P.destroy()
    b.destroy()
    xv.destroy()

    return result


def main():

    ns = [8, 16, 32]  # Mesh ladder
    case = 1  # Which degree ladder to run, see CASES
    degrees = CASES[case]  # Degree ladder, fixed across the sweep

    use_gpu = False

    if use_gpu:
        mat_type = "aijcusparse"
        device = "gpu"
    else:
        mat_type = "aij"
        device = "cpu"

    log_dir = Path("logs")
    if MPI.COMM_WORLD.rank == 0:
        log_dir.mkdir(exist_ok=True)

    smoother = "hiptmair"
    smoother_its = 3
    rtol = 1e-10
    max_it = 1000
    monitor = False

    comm = MPI.COMM_WORLD

    results = []

    PETSc.Log.begin()

    for n in ns:
        par_print(comm, f"\n=== n = {n}, degrees = {degrees} ===")

        stage = PETSc.Log.Stage(f"n={n}")
        stage.push()
        results.append(
            solve(
                comm, n, degrees, mat_type, smoother, smoother_its, rtol, max_it, monitor
            )
        )
        stage.pop()

        r = results[-1]
        par_print(
            comm,
            f"ndofs = {r['ndofs']}, its = {r['its']}, "
            f"curl L2 = {r['curl_l2']:.4e}",
        )

    log_file = log_dir / f"{device}_sweep_{case}.log"
    with redirect_stdout_to(str(log_file), comm):
        for r in results:
            par_print(
                comm,
                f"=== n = {r['n']}, degrees = {degrees}, ranks = {comm.size} ===",
            )
            par_print(comm, f"ndofs        = {r['ndofs']}")
            par_print(comm, f"iterations   = {r['its']}")
            par_print(comm, f"curl L2 error= {r['curl_l2']:.6e}")
            par_print(comm, "")
        dolfinx.common.list_timings(comm)
        PETSc.Log.view()

    header = (
        f"{'n':>5} {'ndofs':>12} {'its':>5} "
        f"{'curl error':>12} {'rate':>6}"
    )
    lines = [
        f"Sweep summary (degrees = {degrees}, smoother = {smoother}, ranks = {comm.size})",
        header,
        "-" * len(header),
    ]

    for i, r in enumerate(results):
        if i == 0:
            curl_rate = float("nan")
        else:
            prev = results[i - 1]
            h_ratio = np.log(prev["n"] / r["n"])
            curl_rate = np.log(r["curl_l2"] / prev["curl_l2"]) / h_ratio
        lines.append(
            f"{r['n']:>5} {r['ndofs']:>12} {r['its']:>5} "
            f"{r['curl_l2']:>12.4e} {curl_rate:>6.2f}"
        )

    par_print(comm, "\n" + "\n".join(lines))

if __name__ == "__main__":
    main()
