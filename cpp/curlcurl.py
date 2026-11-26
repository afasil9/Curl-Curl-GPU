"""Curl-curl + mass bilinear form on degree-1 Nedelec, for the assembled AMS coarse matrix.

FFCx compiles this file into curlcurl.c / curlcurl.h (see CMakeLists.txt). The element must
match the one main.cpp uses for level 0 (N1E, degree 1, legendre variant), otherwise
dolfinx::fem::create_form rejects the function space.
"""

import basix
from basix.ufl import element
from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, curl, dx, inner
from ufl import Coefficient, SpatialCoordinate, as_vector, pi, sin

e = element("N1curl", "tetrahedron", 1, lagrange_variant=basix.LagrangeVariant.legendre)
mesh = Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(curl(u), curl(v)) * dx + inner(u, v) * dx

e3 = element("N1curl", "tetrahedron", 3, lagrange_variant=basix.LagrangeVariant.legendre)
V3 = FunctionSpace(mesh, e3)
v3 = TestFunction(V3)

x = SpatialCoordinate(mesh)
u_ex = as_vector((sin(pi*x[1])*sin(pi*x[2]),
                  sin(pi*x[0])*sin(pi*x[2]),
                  sin(pi*x[0])*sin(pi*x[1])))
f = (2*pi**2 + 1) * u_ex

L = inner(f, v3) * dx                                     # load vector b_i = ∫ f·φ_i

uh = Coefficient(V3)                                      # the computed solution
M_L2   = inner(uh - u_ex, uh - u_ex) * dx                 # ‖u_h − u‖²
M_curl = inner(curl(uh - u_ex), curl(uh - u_ex)) * dx     # ‖curl(u_h − u)‖²

# Forms for FFCx to compile (names other than a, L, M are only picked up when listed here)
forms = [a, L, M_L2, M_curl]
