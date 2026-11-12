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


double determinant(std::vector<std::vector<double>> J)
{
    double detJ;

    detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
           J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
           J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    return detJ;
}

void inverse(const std::vector<std::vector<double>> J, std::vector<std::vector<double>> &Jinv)
{
    double detJ = determinant(J);



    Jinv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) / detJ;
    Jinv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) / detJ;
    Jinv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / detJ;

    Jinv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) / detJ;
    Jinv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / detJ;
    Jinv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) / detJ;

    Jinv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / detJ;
    Jinv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) / detJ;
    Jinv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) / detJ;
}


void piola_map(auto msh){
    auto x = msh->geometry().x(); // Coordiantes of mesh verts x = [x0, y0, z0, x1, y1, z1, ...]
    auto v = msh->geometry().dofmaps().front();
    size_t n_vert = msh->topology()->index_map(0)->size_global();
    size_t n_cells = msh->topology()->index_map(3)->size_local();;

    auto node0 = v(0, 0);
    auto node1 = v(0, 1);
    auto node2 = v(0, 2);
    auto node3 = v(0, 3);

    std::array<double, 3> e0, e1, e2;
    // std::cout << "cell 0 nodes: " << node0 << " " << node1 << " " << node2 << " " << node3 << std::endl;
    // std::cout << "node1 coords: " << x[3*node1 + 0] << " " << x[3*node1 + 1] << " " << x[3*node1 + 2] << std::endl;
    // std::cout << "node0 coords: " << x[3*node0 + 0] << " " << x[3*node0 + 1] << " " << x[3*node0 + 2] << std::endl;

    // std::cout << "size of x is " << x.size() << std::endl;

    std::vector<std::vector<std::vector<double>>> J(n_cells, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0)));
    std::vector<double> detj(n_cells, 0.0);
    std::vector<std::vector<std::vector<double>>> Jinv(n_cells, std::vector<std::vector<double>>(3, std::vector<double>(3, 0.0)));

    // e0 (x1 - x0, y1 - y0, z1 - z0)
    e0 = {x[3*node1 + 0] - x[3*node0 + 0], x[3*node1 + 1] - x[3*node0 + 1], x[3*node1 + 2] - x[3*node0 + 2]};
    e1 = {x[3*node2 + 0] - x[3*node0 + 0], x[3*node2 + 1] - x[3*node0 + 1], x[3*node2 + 2] - x[3*node0 + 2]};

    for (int cell = 0; cell < n_cells; cell++){
        auto node0 = v(cell, 0);
        auto node1 = v(cell, 1);
        auto node2 = v(cell, 2);
        auto node3 = v(cell, 3);
        for (int edge = 0; edge < 3; edge++){
            e0[edge] = x[3*node1 + edge] - x[3*node0 + edge];
            e1[edge] = x[3*node2 + edge] - x[3*node0 + edge];
            e2[edge] = x[3*node3 + edge] - x[3*node0 + edge];
        }
        // J[cell][row][col] = e[row][col]

        //         |  e0[0]  e1[0]  e2[0]  |   |  x1-x0  x2-x0  x3-x0  |
        // J     = |  e0[1]  e1[1]  e2[1]  | = |  y1-y0  y2-y0  y3-y0  |
        //         |  e0[2]  e1[2]  e2[2]  |   |  z1-z0  z2-z0  z3-z0  |

        J[cell][0][0] = e0[0];
        J[cell][1][0] = e0[1];
        J[cell][2][0] = e0[2];

        J[cell][0][1] = e1[0];
        J[cell][1][1] = e1[1];
        J[cell][2][1] = e1[2];

        J[cell][0][2] = e2[0];
        J[cell][1][2] = e2[1];
        J[cell][2][2] = e2[2];
    }


    std::cout << "DetJ array is" << detj << std::endl;
    for (int c = 0; c < n_cells; c++){
        detj[c] = std::abs(determinant(J[c]));
        inverse(J[c], Jinv[c]);
    }
    // std::cout << J << std::endl;
    // std::cout << "\n" << std::endl;

    // for (std::size_t c = 0; c < J.size(); c++) {
    // std::cout << "cell " << c << ": " << J[c] << "\n";
// }
    
    std::cout << "det(J) for cell 0: " << determinant(J[0]) << "\n";

    for (std::size_t c = 0; c < detj.size(); c++) {
        // std::cout << "det(J) for cell " << c << ": " << detj[c] << "\n";
        std::cout << "Jinv for cell " << c << ": " << Jinv[c] << "\n";
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
