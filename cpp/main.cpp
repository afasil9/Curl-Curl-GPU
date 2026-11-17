// ---- Standard library -------------------------------------------------------
#include <algorithm> // std::sort, std::unique, std::lower_bound (permutation_classes)
#include <array>     // std::array (curls, basis values, edge vectors)
#include <cmath>     // std::abs (piola_map)
#include <cstdint>   // std::int32_t, std::uint32_t (class indices, permutation codes)
#include <iostream>  // std::cout
#include <memory>    // std::make_shared (mesh)
#include <span>      // std::span (dense_transform)
#include <vector>

// ---- MPI --------------------------------------------------------------------
#include <mpi.h>

// ---- Basix: reference-element data ------------------------------------------
#include <basix/cell.h>             // basix::cell::type::tetrahedron
#include <basix/element-families.h> // element::family::N1E, lagrange_variant
#include <basix/finite-element.h>   // create_element, tabulate, T_apply, ...
#include <basix/mdspan.hpp>         // basix::impl::mdspan_t (view over the tabulation)
#include <basix/polyset.h>          // polyset::type::standard (needed by make_quadrature)
#include <basix/quadrature.h>       // quadrature::make_quadrature

// ---- DOLFINx: mesh, function space, dofmap ----------------------------------
#include <dolfinx/common/log.h>      // init_logging (required before any dolfinx call)
#include <dolfinx/fem/FiniteElement.h> // wraps the Basix element for DOLFINx
#include <dolfinx/fem/FunctionSpace.h> // FunctionSpace, dofmap
#include <dolfinx/fem/functionspace_factory.h> // create_functionspace
#include <dolfinx/fem/DirichletBC.h>   // locate_dofs_topological (boundary dofs)
#include <dolfinx/mesh/utils.h>        // exterior_facet_indices
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h> // create_box (unit cube of tets)

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
        // Six curl-curl coefficients
        coeffs[12 * c + 0] = H[id3(c, 0, 0)] / detj[c]; // H00
        coeffs[12 * c + 1] = H[id3(c, 1, 1)] / detj[c]; // H11
        coeffs[12 * c + 2] = H[id3(c, 2, 2)] / detj[c]; // H22
        coeffs[12 * c + 3] = H[id3(c, 0, 1)] / detj[c]; // H01 = H10
        coeffs[12 * c + 4] = H[id3(c, 0, 2)] / detj[c]; // H02 = H20
        coeffs[12 * c + 5] = H[id3(c, 1, 2)] / detj[c]; // H12 = H21

        // Six mass coefficients
        coeffs[12 * c + 6]  = detj[c] * G[id3(c, 0, 0)]; // G00
        coeffs[12 * c + 7]  = detj[c] * G[id3(c, 1, 1)]; // G11
        coeffs[12 * c + 8]  = detj[c] * G[id3(c, 2, 2)]; // G22
        coeffs[12 * c + 9]  = detj[c] * G[id3(c, 0, 1)]; // G01 = G10
        coeffs[12 * c + 10] = detj[c] * G[id3(c, 0, 2)]; // G02 = G20
        coeffs[12 * c + 11] = detj[c] * G[id3(c, 1, 2)]; // G12 = G21
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

    std::vector<double> K(6 * ndofs * ndofs, 0.0); // Stiffness matrix
    std::vector<double> M(6 * ndofs * ndofs, 0.0); // Mass matrix
    
    // Symmetric component pairs (a, b); matches SYM in basic_ops.py
    constexpr int SYM[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};

    //Curl part of the reference matrix
    for (std::size_t q = 0; q < nq; ++q)
    {
        const double w = wts[q];
        for (std::size_t basis_i = 0; basis_i < ndofs; ++basis_i)
        {
            // Curl of basis function i (depends only on q, basis_i)
            const std::array<double, 3> curl_i = {
                T(2, q, basis_i, 2) - T(3, q, basis_i, 1),  // ∂y φ_i,z - ∂z φ_i,y
                T(3, q, basis_i, 0) - T(1, q, basis_i, 2),  // ∂z φ_i,x - ∂x φ_i,z
                T(1, q, basis_i, 1) - T(2, q, basis_i, 0)}; // ∂x φ_i,y - ∂y φ_i,x
            
            const std::array<double, 3> phi_i = {
                T(0, q, basis_i, 0),
                T(0, q, basis_i, 1),
                T(0, q, basis_i, 2)
            };


            for (std::size_t basis_j = 0; basis_j < ndofs; ++basis_j)
            {
                // Curl of basis function j (depends only on q, basis_j)
                const std::array<double, 3> curl_j = {
                    T(2, q, basis_j, 2) - T(3, q, basis_j, 1),  // ∂y φ_j,z - ∂z φ_j,y
                    T(3, q, basis_j, 0) - T(1, q, basis_j, 2),  // ∂z φ_j,x - ∂x φ_j,z
                    T(1, q, basis_j, 1) - T(2, q, basis_j, 0)}; // ∂x φ_j,y - ∂y φ_j,x
                const std::array<double, 3> phi_j = {
                    T(0, q, basis_j, 0),
                    T(0, q, basis_j, 1),
                    T(0, q, basis_j, 2)
                };
                for (int k = 0; k < 6; ++k)
                {
                    const int a = SYM[k][0];
                    const int b = SYM[k][1];

                    double val_curl = curl_i[a] * curl_j[b];
                    double val_mass = phi_i[a] * phi_j[b];
                    if (a != b){
                        val_curl += curl_i[b] * curl_j[a];
                        val_mass += phi_i[b] * phi_j[a];
                    }

                    K[k * ndofs * ndofs + basis_i * ndofs + basis_j] += w * val_curl;
                    M[k * ndofs * ndofs + basis_i * ndofs + basis_j] += w * val_mass;
                }
            }
        }
    }

    std::vector<double> A;
    A.reserve(K.size() + M.size());
    A.insert(A.end(), K.begin(), K.end());  
    A.insert(A.end(), M.begin(), M.end());

    return A;
}

// Group cells by their permutation code (same as np.unique(perm, return_inverse=True)).
//   uniq[k] = the k-th distinct permutation code
//   clas[c] = k  means cell c has code uniq[k], so it uses folded reference matrix k

struct permutation_class {
    std::vector<std::int32_t> clas; // class index for each cell
    std::vector<std::uint32_t> uniq; // unique permutation codes
};

permutation_class permutation_classes(auto msh)
{
    msh->topology_mutable()->create_cell_permutations();
    const std::vector<std::uint32_t>& perm = msh->topology()->get_cell_permutation_info();
    const std::size_t n_cells = msh->topology()->index_map(3)->size_local();

    std::vector<std::uint32_t> uniq(perm.begin(), perm.begin() + n_cells);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

    std::vector<std::int32_t> clas(n_cells);
    for (std::size_t c = 0; c < n_cells; ++c)
        clas[c] = std::lower_bound(uniq.begin(), uniq.end(), perm[c]) - uniq.begin();

    return {clas, uniq};
}

std::vector<double> dense_transform(const basix::FiniteElement<double>& element,
                                    const std::vector<std::uint32_t>& uniq,
                                    bool transpose)
{
    int nd = element.dim();
    int n_class = uniq.size();

    std::vector<double> out(n_class * nd * nd, 0.0);

    for (int p = 0; p < n_class; ++p){
        for (int i = 0 ; i< nd; ++i){
            out[p * nd * nd +  i * nd + i] = 1.0;
        }
        std::span<double> block(out.data() + p * nd * nd, nd * nd);

        if (transpose)
            element.Tt_apply(block, nd, uniq[p]);
        else
            element.T_apply(block, nd, uniq[p]);
    }    
    return out;
}


std::vector<double> fold_reference(const std::vector<double>& refs,
                                   const std::vector<double>& T,
                                   const std::vector<double>& Tt,
                                   int n_class, int nd)
{

    // folded[p][k][i][j]  =  folded[((p * 12 + k) * nd + i) * nd + j]
    // p perm class, k = 0..5 for curl-curl then 6..11 for mass, i = Test basis functions, j = Trial basis functions
    std::vector<double> folded(n_class * 12 * nd * nd, 0.0); // result: (n_class, 12, nd, nd)
    std::vector<double> tmp(nd * nd, 0.0);                        // scratch: one nd x nd matrix



    for (int k = 0; k < 12; ++k){ // Loop over 6 curl then 6 mass
        for (int p = 0; p < n_class; ++p){
            // tmp = T_p · R[k]
            std::fill(tmp.begin(), tmp.end(), 0.0);
            for (int rows_tp = 0; rows_tp < nd; ++rows_tp){
                for (int cols_rk = 0; cols_rk < nd; ++cols_rk){
                    for (int m = 0; m < nd; ++m){
                        // Tp [i][m] = T[p * nd * nd + i * nd + m]
                        // Rk [m][j] = refs[k * nd * nd + m * nd + j]
                        // tmp[i][j] = Σₘ Tp[i][m] * Rk[m][j]
                        tmp[rows_tp * nd + cols_rk] += T[p * nd * nd + rows_tp * nd + m] *
                                                       refs[k * nd * nd + m * nd + cols_rk];
                    }
                }
            }

            // folded[p][k] = tmp · Tt_p  (same p and k, while tmp still holds T_p · R[k])
            for (int rows_tp = 0; rows_tp < nd; ++rows_tp){
                for (int cols_rk = 0; cols_rk < nd; ++cols_rk){
                    for (int m = 0; m < nd; ++m){
                        // tmp [i][m] = tmp[i * nd + m]
                        // Tt [m][j] = Tt[p * nd * nd + m * nd + j]
                        // folded[p][k][i][j] = Σₘ tmp[i][m] * Tt[m][j]
                        folded[((p * 12 + k) * nd + rows_tp) * nd + cols_rk] +=
                            tmp[rows_tp * nd + m] * Tt[p * nd * nd + m * nd + cols_rk];
                    }
                }
            }
        }
    }
    return folded;
}


std::vector<double> matvec(const std::vector<double>& x, const std::vector<double>& coeffs, const permutation_class& pc, int nd, const std::vector<double>& folded, auto dofmap)
{

    //dofmap = (n_cells, nd)
    const std::size_t n_cells = pc.clas.size();
    std::vector<double> y_vec(x.size(), 0.0);

    for (std::size_t c = 0; c < n_cells; ++c){
        int p = pc.clas[c]; // permutation class of cell c
        for (int k = 0; k < 12; ++k){ // Loop over 6 curl then 6 mass
            for (int i = 0; i < nd; ++i){ // Test basis functions
                double sum = 0.0;
                for (int j = 0; j < nd; ++j){ // Trial basis functions
                    sum += folded[((p * 12 + k) * nd + i) * nd + j] * x[(dofmap(c, j))];
                }
                y_vec[dofmap(c, i)] += coeffs[12 * c + k] * sum;
            }
        }
    }
    return y_vec;
    }


std::vector<double> getdiagonal(const std::vector<double>& coeffs, const permutation_class& pc, int nd, const std::vector<double>& folded, auto dofmap, int ndofs)
{
    const std::size_t n_cells = pc.clas.size();
    std::vector<double> diagonal(ndofs, 0.0);

    for (std::size_t c = 0; c < n_cells; ++c){
        int p = pc.clas[c]; // permutation class of cell c
        for (int k = 0; k < 12; ++k){ // Loop over 6 curl then 6 mass
            for (int i = 0; i < nd; ++i){ // Test basis functions
                double sum = 0.0;
                sum += folded[((p * 12 + k) * nd + i) * nd + i];
                diagonal[dofmap(c, i)] += coeffs[12 * c + k] * sum;
            }
        }
    }
    return diagonal;
}

// Boundary mask: bc_mask[g] = 1.0 if global dof g lies on the boundary, else 0.0.
std::vector<double> boundary_dof_mask(auto msh, auto V, std::int32_t ndofs)
{
    const int tdim = msh->topology()->dim();
    auto topology = msh->topology_mutable();
    topology->create_entities(tdim - 1);          // facets
    topology->create_connectivity(tdim - 1, tdim); // facet -> cell (to find exterior facets)
    topology->create_connectivity(tdim, tdim - 1); // cell -> facet (to find dofs on facets)

    const std::vector<std::int32_t> facets = dolfinx::mesh::exterior_facet_indices(*topology);
    const std::vector<std::int32_t> bdofs = dolfinx::fem::locate_dofs_topological(
        *topology, *V->dofmap(), tdim - 1, std::span<const std::int32_t>(facets));

    std::vector<double> bc_mask(ndofs, 0.0);
    // Mark the boundary dofs as 1.0 in mask. bc_mask[g] = 1.0 if dof g lies on a boundary edge or face
    for (std::int32_t g : bdofs)
        if (g < ndofs)
            bc_mask[g] = 1.0;
    return bc_mask;
}


std::vector<double> matvec_bc(const std::vector<double>& x,
                              const std::vector<double>& bc_mask,
                              const std::vector<double>& coeffs,
                              const permutation_class& pc, int nd,
                              const std::vector<double>& folded, auto dofmap)
{
    std::vector<double> x_in = x;

    for (std::size_t g = 0; g < x.size(); ++g)
    if (bc_mask[g] == 1.0)
        x_in[g] = 0.0; // Zero out the boundary dofs in the input vector
    std::vector<double> y = matvec(x_in, coeffs, pc, nd, folded, dofmap);

    for (std::size_t g = 0; g < y.size(); ++g)
        if (bc_mask[g] == 1.0)
        y[g] = x[g];

    return y;
}

std::vector<double> diagonal_bc(const std::vector<double>& bc_mask,
                                const std::vector<double>& coeffs,
                                const permutation_class& pc, int nd,
                                const std::vector<double>& folded, auto dofmap, int ndofs)
{
    std::vector<double> d = getdiagonal(coeffs, pc, nd, folded, dofmap, ndofs);

    for (std::size_t g = 0; g < d.size(); ++g)
    {
        if (bc_mask[g] == 1.0)
            d[g] = 1.0; // boundary row is all zeros except a 1 on the diagonal
    }
    return d;
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    dolfinx::init_logging(argc, argv);
    {
        auto msh = std::make_shared<dolfinx::mesh::Mesh<double>>(
            dolfinx::mesh::create_box<double>(
                MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {4, 4, 4},
                dolfinx::mesh::CellType::tetrahedron));

        const int tdim = msh->topology()->dim();
        // std::cout << "Mesh: " << msh->topology()->index_map(tdim)->size_global() << " cells, "
                //   << msh->topology()->index_map(0)->size_global() << " vertices" << std::endl;

        std::vector<double> coeffs = piola_map(msh); // (n_cells, 12): 6 curl-curl then 6 mass
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

        permutation_class pc = permutation_classes(msh);
        std::cout << "permutation classes: " << pc.uniq.size() << "\n";

        std::vector<double> Tdense = dense_transform(element, pc.uniq, false);
        std::vector<double> Ttdense = dense_transform(element, pc.uniq, true);

        std::vector<double> folded = fold_reference(refs, Tdense, Ttdense, pc.uniq.size(), nd);


        auto fe = std::make_shared<const dolfinx::fem::FiniteElement<double>>(element, 3);
        auto V  = std::make_shared<dolfinx::fem::FunctionSpace<double>>(
                    dolfinx::fem::create_functionspace<double>(msh, fe));
        auto dofmap = V->dofmap()->map();
        const std::int32_t ndofs = V->dofmap()->index_map->size_local();

        std::vector<double> x(ndofs, 1.0);   // test input: all ones for now

        std::vector<double> y_vec = matvec(x, coeffs, pc, nd, folded, dofmap);

        std::vector<double> diag = getdiagonal(coeffs, pc, nd, folded, dofmap, ndofs);
        



        // std::cout << "refs shape: (12, " << nd << ", " << nd << ")\n";
        // std::cout << "R[0, 0, 0] = " << refs[0] << ", R[6, 0, 0] = " << refs[6 * nd * nd] << "\n";
    }
    MPI_Finalize();
    return 0;
}
