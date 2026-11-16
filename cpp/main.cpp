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
#include <basix/cell.h>             // basix::cell::type::tetrahedron
#include <basix/element-families.h> // element::family::N1E, lagrange_variant
#include <basix/finite-element.h>   // create_element, tabulate, T_apply, ...
// #include <basix/interpolation.h>    // compute_interpolation_operator
// #include <basix/mdspan.hpp>         // multi-dimensional views over flat arrays
#include <basix/polyset.h>          // polyset::type::standard (needed by make_quadrature)
#include <basix/quadrature.h>       // quadrature::make_quadrature

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

inline std::size_t id4(std::size_t c, int i, int j, int k) { return 27 * c + 9 * i + 3 * j + k; }

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


std::vector<double> piola_map(auto msh){
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
    for (std::size_t c = 0; c < n_cells; ++c)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    H[id3(c, i, j)] += J[id3(c, k, i)] * J[id3(c, k, j)];
        

    // std::cout << "J for cell 0:\n";
    // print_mat3(&J[9 * 0]);
    // std::cout << "det(J) for cell 0: " << detj[0] << "\n";
    // std::cout << "Jinv for cell 0:\n";
    // print_mat3(&Jinv[9 * 0]);
    // std::cout << "G for cell 0:\n";
    // print_mat3(&G[9 * 0]);
    // std::cout << "H for cell 0:\n";
    // print_mat3(&H[9 * 0]);

    std::vector<double> coeffs(n_cells * 12, 0.0);


    for (size_t c =0; c< n_cells; c++){
        // Six mass coefficients
        coeffs[12 * c + 0] = detj[c] * G[id3(c, 0, 0)]; // G00
        coeffs[12 * c + 1] = detj[c] * G[id3(c, 1, 1)]; // G11
        coeffs[12 * c + 2] = detj[c] * G[id3(c, 2, 2)]; // G22
        coeffs[12 * c + 3] = detj[c] * G[id3(c, 0, 1)]; // G01 = G10
        coeffs[12 * c + 4] = detj[c] * G[id3(c, 0, 2)]; // G02 = G20
        coeffs[12 * c + 5] = detj[c] * G[id3(c, 1, 2)]; // G12 = G21

        // Six curl-curl coefficients
        coeffs[12 * c + 6]  = H[id3(c, 0, 0)] / detj[c]; // H00
        coeffs[12 * c + 7]  = H[id3(c, 1, 1)] / detj[c]; // H11
        coeffs[12 * c + 8]  = H[id3(c, 2, 2)] / detj[c]; // H22
        coeffs[12 * c + 9]  = H[id3(c, 0, 1)] / detj[c]; // H01 = H10
        coeffs[12 * c + 10] = H[id3(c, 0, 2)] / detj[c]; // H02 = H20
        coeffs[12 * c + 11] = H[id3(c, 1, 2)] / detj[c]; // H12 = H21
    }
    return coeffs;
}

std::vector<double> curlcurl_reference_matrix(const basix::FiniteElement<double>& element,
                                              int quadrature_degree = 2)
{
    constexpr std::size_t tdim = 3;

    auto [pts, wts] = basix::quadrature::make_quadrature<double>(
        basix::quadrature::type::Default, element.cell_type(),
        basix::polyset::type::standard, quadrature_degree);
    const std::size_t nq = wts.size();

    auto [tab, shape] = element.tabulate(1, pts, {nq, tdim});
    const std::size_t ndofs = shape[2];
    const std::size_t vs = shape[3]; // value size = 3 for N1E
    
    basix::impl::mdspan_t<const double, 4> T(tab.data(), shape);

    // T (derivative, quadrature point, basis, value compnent)

    // 
    // Kᵢⱼ = Σ₍q₌₁₎ⁿᵠ wᵩ (∇ × φᵢ)(xᵩ) · (∇ × φⱼ)(xᵩ)
    // 

    double a = T(0, 0, 0 , 0);
    double b = T(1, 0, 0 , 0);
    double c = T(2, 0, 0 , 0);

    std::cout << "T(0, 0, 0, 0) = " << a << ", T(1, 0, 0, 0) = " << b << ", T(2, 0, 0, 0) = " << c << "\n";


    double phi_x = T(0, 0, 0, 0); // x component of basis funciton 0 at point 0
    double phi_y = T(0, 0, 0, 1);
    double phi_z = T(0, 0, 0, 2);
    
    std::vector<std::array<double, 3>> curls(nq * ndofs);

    // T(0, q , 0, 2) gives you z component of basis funciton 0.
    // std::cout << "phi_x = " << phi_x << ", phi_y = " << phi_y << ", phi_z = " << phi_z << "\n";
    // q - quadrature point index, i - basis function index
    
    for (std::size_t q = 0; q < nq; ++q)
    {
        for (std::size_t basis_i = 0; basis_i < ndofs; ++basis_i)
        {
            // const double phi_x = T(0, q, i, 0); 
            // const double phi_y = T(0, q, i, 1);
            // const double phi_z = T(0, q, i, 2);

            const double curl_x = T(2, q, basis_i, 2) - T(3, q, basis_i, 1); // ∂y φ_i,z - ∂z φ_i,y
            const double curl_y = T(3, q, basis_i, 0) - T(1, q, basis_i, 2); // ∂z φ_i,x - ∂x φ_i,z
            const double curl_z = T(1, q, basis_i, 1) - T(2, q, basis_i, 0); // ∂x φ_i,y - ∂y φ_i,x
            // std::cout << "q=" << q << ", i=" << i
            //         << ": phi_x=" << phi_x
            //         << ", phi_y=" << phi_y
            //         << ", phi_z=" << phi_z << "\n";
            curls[q * ndofs + basis_i] = {{curl_x, curl_y, curl_z}};
        }
        const double w = wts[q];
    }

    return std::vector<double>(ndofs * ndofs * 12, 0.0); // Placeholder for the actual reference matrix
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

        std::vector<double> coeffs = piola_map(msh); // (n_cells, 12): 6 mass then 6 curl-curl
        const std::size_t n_cells = coeffs.size() / 12;
        // std::cout << "coeffs shape: (" << n_cells << ", 12)\n";
        // std::cout << "cell 0 coeffs: ";
        for (int k = 0; k < 12; ++k) std::cout << coeffs[k] << " ";
        std::cout << "\n";

        const int degree = 3;
        auto element = basix::create_element<double>(
            basix::element::family::N1E, basix::cell::type::tetrahedron, degree,
            basix::element::lagrange_variant::legendre, basix::element::dpc_variant::unset, false);
        std::vector<double> refs = curlcurl_reference_matrix(element, 2 * degree + 2);
        const std::size_t nd = element.dim();
        // std::cout << "refs shape: (12, " << nd << ", " << nd << ")\n";
        // std::cout << "R[0, 0, 0] = " << refs[0] << ", R[6, 0, 0] = " << refs[6 * nd * nd] << "\n";
    }
    MPI_Finalize();
    return 0;
}
