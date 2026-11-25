"""Curl-curl + mass bilinear form on degree-1 Nedelec, for the assembled AMS coarse matrix.

FFCx compiles this file into curlcurl.c / curlcurl.h (see CMakeLists.txt). The element must
match the one main.cpp uses for level 0 (N1E, degree 1, legendre variant), otherwise
dolfinx::fem::create_form rejects the function space.
"""

import basix
from basix.ufl import element
from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, curl, dx, inner

e = element("N1curl", "tetrahedron", 1, lagrange_variant=basix.LagrangeVariant.legendre)
mesh = Mesh(element("Lagrange", "tetrahedron", 1, shape=(3,)))
V = FunctionSpace(mesh, e)

u = TrialFunction(V)
v = TestFunction(V)

a = inner(curl(u), curl(v)) * dx + inner(u, v) * dx
