// Host-side driver: builds all the tables with Basix/DOLFINx on the CPU,
// then hands plain arrays to the CUDA kernels declared in kernels.hpp.
// No CUDA headers in this file -- it is compiled by the host C++ compiler.

// ---- Standard library -------------------------------------------------------
#include <algorithm> // std::sort, std::unique (permutation_classes)
#include <array>
#include <cmath>
#include <cstdint> // std::int32_t, std::uint32_t (dofmaps, permutation codes)
#include <cstdio>  // printf
#include <iostream>
#include <memory> // std::shared_ptr (mesh, function spaces)
// #include <numeric>
// #include <random> // random start vector for the power iteration
#include <span>
#include <vector>

// ---- MPI --------------------------------------------------------------------
#include <mpi.h>

// ---- Basix: reference-element data ------------------------------------------
// (commented out to keep compiles fast -- uncomment each one when you first need it)
// #include <basix/cell.h>             // basix::cell::type::tetrahedron
// #include <basix/element-families.h> // element::family::N1E, lagrange_variant
// #include <basix/finite-element.h>   // create_element, tabulate, T_apply, ...
// #include <basix/interpolation.h>    // compute_interpolation_operator
// #include <basix/mdspan.hpp>         // multi-dimensional views over flat arrays
// #include <basix/quadrature.h>       // quadrature::make_quadrature

// ---- DOLFINx: mesh, dofmaps, boundary dofs, assembly ------------------------
// #include <dolfinx/fem/DirichletBC.h>   // locate_dofs_topological, DirichletBC
// #include <dolfinx/fem/DofMap.h>        // cell -> dof map, index_map
// #include <dolfinx/fem/FiniteElement.h> // T_apply / Tt_apply / Tt_inv_apply
// #include <dolfinx/fem/Function.h>
// #include <dolfinx/fem/FunctionSpace.h>
// #include <dolfinx/fem/assembler.h> // assemble_vector, assemble_matrix
// #include <dolfinx/fem/utils.h>     // create_functionspace, create_form
// #include <dolfinx/la/MatrixCSR.h>  // coarse (degree 1) matrix
// #include <dolfinx/la/SparsityPattern.h>
// #include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h> // create_box (unit cube of tets)
// #include <dolfinx/mesh/utils.h>      // exterior_facet_indices

// ---- FFCx-generated forms (from forms.py, via `ffcx forms.py`) --------------
// #include "forms.h"      // TODO: enable once forms.py exists and ffcx has been run

// ---- Our CUDA kernels (implemented in kernels.cu) ---------------------------
// #include "kernels.hpp"  // TODO: enable once kernels.hpp / kernels.cu exist

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (std::size_t i = 0; i < vec.size(); i++) {
        os << vec[i];
        if (i + 1 < vec.size()) os << ", ";
    }
    os << "]";
    return os;
}

// All per-cell 3x3 matrices are stored row-major in one flat array:
//     M has shape (n_cells, 3, 3)  ->  M[c, i, j] = M[9*c + 3*i + j]
// so the 9 entries of cell c start at &M[9*c], and inside that block
// entry (i, j) is at [3*i + j]:
//     | [0] [1] [2] |
//     | [3] [4] [5] |
//     | [6] [7] [8] |
inline std::size_t id3(std::size_t c, int i, int j) { return 9 * c + 3 * i + j; }

// Determinant of one 3x3 row-major matrix (M points at its 9 entries).
double determinant(const double* M)
{
    return M[0] * (M[4] * M[8] - M[5] * M[7]) -
           M[1] * (M[3] * M[8] - M[5] * M[6]) +
           M[2] * (M[3] * M[7] - M[4] * M[6]);
}

// Inverse of one 3x3 row-major matrix: writes 9 entries to Minv.
void inverse(const double* M, double* Minv)
{
    const double detM = determinant(M);

    Minv[0] = (M[4] * M[8] - M[5] * M[7]) / detM;
    Minv[1] = (M[2] * M[7] - M[1] * M[8]) / detM;
    Minv[2] = (M[1] * M[5] - M[2] * M[4]) / detM;

    Minv[3] = (M[5] * M[6] - M[3] * M[8]) / detM;
    Minv[4] = (M[0] * M[8] - M[2] * M[6]) / detM;
    Minv[5] = (M[2] * M[3] - M[0] * M[5]) / detM;

    Minv[6] = (M[3] * M[7] - M[4] * M[6]) / detM;
    Minv[7] = (M[1] * M[6] - M[0] * M[7]) / detM;
    Minv[8] = (M[0] * M[4] - M[1] * M[3]) / detM;
}

// Print one 3x3 row-major matrix.
void print_mat3(const double* M)
{
    for (int i = 0; i < 3; i++)
        std::cout << "  [" << M[3 * i + 0] << ", " << M[3 * i + 1] << ", " << M[3 * i + 2] << "]\n";
}


void piola_map(auto msh){
    auto x = msh->geometry().x(); // Coordinates of mesh verts x = [x0, y0, z0, x1, y1, z1, ...], shape (n_nodes, 3)
    auto v = msh->geometry().dofmaps().front(); // v(cell, k) = node index of vertex k of cell, shape (n_cells, 4)
    std::size_t n_cells = msh->topology()->index_map(3)->size_local();

    std::vector<double> J(n_cells * 9, 0.0);    // (n_cells, 3, 3)
    std::vector<double> Jinv(n_cells * 9, 0.0); // (n_cells, 3, 3)
    std::vector<double> detj(n_cells, 0.0);     // (n_cells)

    std::array<double, 3> e0, e1, e2;

    for (std::size_t cell = 0; cell < n_cells; cell++){
        auto node0 = v(cell, 0);
        auto node1 = v(cell, 1);
        auto node2 = v(cell, 2);
        auto node3 = v(cell, 3);
        for (int d = 0; d < 3; d++){
            e0[d] = x[3*node1 + d] - x[3*node0 + d];
            e1[d] = x[3*node2 + d] - x[3*node0 + d];
            e2[d] = x[3*node3 + d] - x[3*node0 + d];
        }

        // J[cell, row, col]: row = coordinate (x, y, z), col = edge (e0, e1, e2)

        //         |  e0[0]  e1[0]  e2[0]  |   |  x1-x0  x2-x0  x3-x0  |
        // J     = |  e0[1]  e1[1]  e2[1]  | = |  y1-y0  y2-y0  y3-y0  |
        //         |  e0[2]  e1[2]  e2[2]  |   |  z1-z0  z2-z0  z3-z0  |

        for (int row = 0; row < 3; row++){
            J[id3(cell, row, 0)] = e0[row];
            J[id3(cell, row, 1)] = e1[row];
            J[id3(cell, row, 2)] = e2[row];
        }
    }

    for (std::size_t c = 0; c < n_cells; c++){
        detj[c] = std::abs(determinant(&J[9 * c]));
        inverse(&J[9 * c], &Jinv[9 * c]);
    }

    // G = einsum("cik,cjk->cij", Jinv, Jinv)   ->   G_c = Jinv_c * Jinv_c^T
    // For every cell c, and for fixed row i and j, we sum over the columns k

    // Mass matrix coefficient.
    std::vector<double> G(n_cells * 9, 0.0);

    for (std::size_t c = 0; c < n_cells; c++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    G[id3(c, i, j)] += Jinv[id3(c, i, k)] * Jinv[id3(c, j, k)];

    // Curl coefficients
    std::vector<double> H(n_cells * 9, 0.0);
    for (std::size_t c = 0; c < n_cells; c++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    H[id3(c, i, j)] += J[id3(c, i, k)] * J[id3(c, j, k)];
    

    // std::cout << "J for cell 0:\n";
    // print_mat3(&J[9 * 0]);
    // std::cout << "det(J) for cell 0: " << detj[0] << "\n";
    // std::cout << "Jinv for cell 0:\n";
    // print_mat3(&Jinv[9 * 0]);
    // std::cout << "G for cell 0:\n";
    // print_mat3(&G[9 * 0]);
    // std::cout << "H for cell 0:\n";
    // print_mat3(&H[9 * 0]);

    std::vector<double> coeffs(n_cells * 18, 0.0);


    for (int c =0; c< n_cells; c++){
        coeffs[18 * c + 0] = detj[c] * G[id3(c, 0, 0)];   // u_x * v_x
        coeffs[18 * c + 1] = detj[c] * G[id3(c, 0, 1)];   // u_x * v_y
        coeffs[18 * c + 2] = detj[c] * G[id3(c, 0, 2)];
        coeffs[18 * c + 3] = detj[c] * G[id3(c, 1, 0)];
        coeffs[18 * c + 4] = detj[c] * G[id3(c, 1, 1)];
        coeffs[18 * c + 5] = detj[c] * G[id3(c, 1, 2)];
        coeffs[18 * c + 6] = detj[c] * G[id3(c, 2, 0)];
        coeffs[18 * c + 7] = detj[c] * G[id3(c, 2, 1)];
        coeffs[18 * c + 8] = detj[c] * G[id3(c, 2, 2)];

        coeffs[18 * c + 9] = H[id3(c, 0, 0)] / detj[c];
        coeffs[18 * c + 10] = H[id3(c, 0, 1)] / detj[c];
        coeffs[18 * c + 11] = H[id3(c, 0, 2)] / detj[c];
        coeffs[18 * c + 12] = H[id3(c, 1, 0)] / detj[c];
        coeffs[18 * c + 13] = H[id3(c, 1, 1)] / detj[c];
        coeffs[18 * c + 14] = H[id3(c, 1, 2)] / detj[c];
        coeffs[18 * c + 15] = H[id3(c, 2, 0)] / detj[c];
        coeffs[18 * c + 16] = H[id3(c, 2, 1)] / detj[c];
        coeffs[18 * c + 17] = H[id3(c, 2, 2)] / detj[c];
    }


}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    {
        auto msh = std::make_shared<dolfinx::mesh::Mesh<double>>(
            dolfinx::mesh::create_box<double>(
                MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {4, 4, 4},
                dolfinx::mesh::CellType::tetrahedron));

        const int tdim = msh->topology()->dim();
        // std::cout << "Mesh: " << msh->topology()->index_map(tdim)->size_global() << " cells, "
                //   << msh->topology()->index_map(0)->size_global() << " vertices" << std::endl;

        piola_map(msh);
    }
    MPI_Finalize();
    return 0;
}
