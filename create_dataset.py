from mpi4py import MPI
from dolfinx import mesh
import ufl
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, locate_dofs_topological, Expression
from dolfinx.fem.petsc import LinearProblem  
from dolfinx import fem, default_scalar_type
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

def generate_data(nsamples, nx):
    uh = torch.zeros((nsamples, nx))
    graduh = torch.zeros((nsamples, nx))
    a = torch.zeros((nsamples, nx))
    epsilons = torch.zeros((nsamples,))
    
    for i in range(nsamples):
        mu, sigma = -2.0, 1.0
        eps = np.random.lognormal(mean=mu, sigma=sigma)
        eps = np.clip(eps, 0.0001, 1.0) 
        
        y_i = np.random.uniform(0.01,1.0)
        
        points, this_uh, this_a, this_graduh = solver(eps, y_i, nx)
        normal = np.max(np.abs(this_uh.x.array))
        
        uh[i, :] = torch.Tensor(this_uh.x.array / normal)
        graduh[i, :] = torch.Tensor(this_graduh.x.array / normal)
        a[i, :] = torch.Tensor(this_a.x.array * normal)
        epsilons[i] = eps
    
    points = torch.Tensor(points)
    return points, a, uh, graduh, epsilons

def solver(eps, y_i, nx):
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, nx - 1)
    V = functionspace(domain, ("Lagrange", 1))
    uD = Constant(domain, default_scalar_type(0))
    
    # Condizioni al contorno
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = locate_dofs_topological(V, fdim, boundary_facets)
    bc = dirichletbc(uD, boundary_dofs, V)
    
    # Definizione di c(x)
    diff_a = Function(V)
    def c_fun(x):
       return 2.0 + y_i * np.sin(np.pi * x[0])

    
    diff_a.interpolate(c_fun)
    
    # Definizione dell'equazione
    f = Constant(domain, default_scalar_type(1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = (eps**2) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + diff_a * u * v * ufl.dx
    L = f * v * ufl.dx
    
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    graduh_f = ufl.grad(uh)
    graduh_exp = Expression(graduh_f, V.element.interpolation_points())
    graduh = Function(V)
    graduh.interpolate(graduh_exp)
    
    points = V.tabulate_dof_coordinates()[:, 0]
    return points, uh, diff_a, graduh

# Numero di campioni e punti spaziali
nsamples = 200  # Cambia secondo necessità
nx = 50  # Risoluzione spaziale per la mesh

# Generazione del dataset
points, a, uh, graduh, epsilons = generate_data(nsamples, nx)

# Salvataggio su file .pt
torch.save({
    "points": points,   # Coordinate spaziali x
    "a": a,             # Coefficiente c(x)
    "uh": uh,           # Soluzione u(x)
    "graduh": graduh,   # Gradiente u'(x)
    "epsilons": epsilons  # Valori di ε
}, "dataset.pt")

print("Dataset salvato come dataset.pt")

