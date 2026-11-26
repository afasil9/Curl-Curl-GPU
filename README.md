# Curl-Curl-GPU

A matrix-free **p-multigrid** solver for the curl-curl + mass problem on Nédélec elements, written so
that it can be moved to the GPU.

The problem is: find **u** ∈ H(curl) such that

```
curl curl u + u = f   in Ω = [0, 1]³
          u × n = 0   on ∂Ω
```

The right-hand side comes from a manufactured solution, so the error can be measured exactly:

```
u_ex = (sin πy sin πz,  sin πx sin πz,  sin πx sin πy)
f    = (2π² + 1) u_ex
```

It is discretised with Nédélec (first kind, `N1curl`) elements on tetrahedra. Every operator, from
the matrix-vector product to the transfers between polynomial degrees and the smoother, is applied
**without assembling a global matrix**. Each one is a loop over cells using small precomputed
reference matrices.

---

## How it works

For each cell `c`, the local matrix is

```
A_c = Σ_k  coeffs[c, k] · folded[p, k]          k = 0 … 11,   p = clas[c]
```

- **`coeffs[c, k]`**: 12 numbers per cell from the covariant Piola map (`piola_map`). Six are
  curl-curl coefficients (`JᵀJ / |det J|`) and six are mass coefficients (`|det J| · J⁻¹J⁻ᵀ`), one
  for each symmetric pair of components. Together they describe how the reference tetrahedron is
  stretched into cell `c`.
- **`folded[p, k]`**: the 12 reference matrices (`curlcurl_reference_matrix`), with the DOF
  transformation for permutation class `p` already applied (`fold_reference`). Cells whose edges
  and faces are oriented the same way share a class (`permutation_classes`), so only one folded
  matrix is stored per class.
- **Transfers:** prolongation interpolates from degree `p` to `p+1` (`build_transfer`,
  `apply_transfer`). It is matrix-free in the same way: one small interpolation matrix is
  precomputed per permutation class, and each cell does a small dense matvec. Restriction is its
  exact transpose (`apply_transfer_transpose`).
- **Hiptmair smoother** on levels 1 and 2. The curl-curl term vanishes on gradients
  (`curl ∇φ = 0`), so gradient error modes only see the mass term. They have very small eigenvalues,
  and edge Jacobi alone damps them very slowly. Each Hiptmair application therefore does:
  ```
  z  = ω D_e⁻¹ r                  edge Jacobi         D_e = diag(A)
  r₂ = r − A z
  z += G (ω D_n⁻¹ Gᵀ r₂)          nodal Jacobi        D_n = diag(Gᵀ A G)
  ```
  with `ω = 0.8`. `G` is the matrix-free discrete gradient from Lagrange P_p to Nédélec degree p
  (`build_gradient`, `apply_G`, `apply_Gt`). `diag(Gᵀ A G)` is computed cell by cell
  (`diag_gtAg`). Hiptmair is the preconditioner inside a Chebyshev smoother.
- **Coarse solve** on level 0: CG preconditioned with **hypre AMS**. This is the only level whose
  matrix is assembled, from a UFL form (`cpp/curlcurl.py`, compiled with FFCx). The same file also
  defines the load vector `b` (degree 3) and the error norms.

### PETSc solver tree

All matrix-free operators are wrapped as PETSc shell objects (`MATSHELL`, `PCSHELL`):

```
outer solver: CG on A[2] u = b
 └─ preconditioner: PCMG (one V-cycle)
      ├─ level 2 smoother: KSP Chebyshev, 2 iterations
      │     └─ preconditioner: Hiptmair_apply
      ├─ level 1 smoother: KSP Chebyshev, 2 iterations
      │     └─ preconditioner: Hiptmair_apply
      └─ level 0 coarse solve: KSP CG
            └─ preconditioner: hypre AMS
```

### Results (4×4×4 mesh, degree 3, manufactured f)

| smoother on levels 1, 2 | outer iterations |
|---|---|
| none (CG + Jacobi, no multigrid) | 2429 |
| Chebyshev + Jacobi | 108 |
| **Chebyshev + Hiptmair** | **12** |

Relative tolerance 1e-8 on the preconditioned residual. Errors against the exact solution:
`‖u − u_h‖_L2 = 3.33e-3`, `‖curl(u − u_h)‖_L2 = 1.52e-2`.

---


## Requirements

- DOLFINx 0.12 (C++ library **and** Python package, including FFCx and Basix)
- PETSc built with **hypre**, which provides the AMS coarse solver
- MPI, CMake ≥ 3.21, and a C++20 compiler

The `ffcx` executable must be on your `PATH` or in `$HOME/Fenics/fenics_env/bin`. Edit the
`find_program` hint in `cpp/CMakeLists.txt` if your environment is elsewhere.

## Build

From `cpp/`, with `P` set to the directory where DOLFINx, Basix, PETSc and SLEPc are installed:

```bash
cd cpp
P=/path/to/packages
PETSC_DIR=$P/petsc cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$P/dolfinx;$P/basix;$P/petsc;$P/slepc"
cmake --build build -j
```

Without `-DCMAKE_BUILD_TYPE=Release`, the project builds in **Debug** mode (`-O0 -g3`, for lldb),
which is much slower. A full solve takes about 1–2 s in Release and about 1 minute in Debug.

After the first configure, rebuild with:

```bash
cmake --build build -j
```

## Run

From `cpp/`:

```bash
./build/main          # 4×4×4 mesh
./build/main -N 8     # 8×8×8 mesh
```

The program prints the size of each level, the outer iteration count, and the L2 and curl errors
against the exact solution:

The program calls `KSPSetFromOptions`, so the solver can be changed from the command line without
recompiling:

| option | effect |
|---|---|
| `-N 8` | number of cells per direction (default 4) |
| `-ksp_monitor` | print the residual every outer iteration |
| `-ksp_view` | print the full solver tree |
| `-mg_coarse_ksp_converged_reason` | AMS coarse-solve iterations in each V-cycle |
| `-mg_levels_ksp_max_it 3` | 3 smoothing steps per level instead of 2 |
| `-mg_levels_pc_type jacobi` | Chebyshev + Jacobi smoothing instead of Hiptmair |
| `-ksp_type cg` | plain CG as the outer solver |
| `-ksp_norm_type unpreconditioned` | apply the tolerance to the true residual `‖b − A u‖` |

The `-ksp_monitor` residual is the *preconditioned* residual, because that is CG's default norm.

---
