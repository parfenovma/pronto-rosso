using Gridap
using Gridap.ODEs
using Gridap.Geometry

L = 3.0
H = 1.0
n = 40
model_ref = CartesianDiscreteModel((0,L,0,H), (3*n, n))

function mapping(x)
    xi, yi = x[1], x[2]
    envelope = 0.1 * sin(2*pi*xi)
    new_y = (yi/H) * (H - 2*envelope) + envelope
    return VectorValue(xi, new_y)
end

base_model = UnstructuredDiscreteModel(model_ref)
model = Gridap.Geometry.MappedDiscreteModel(base_model, mapping)


reffe = ReferenceFE(lagrangian, Float64, 1)

V0 = TestFESpace(model, reffe, conformity=:H1)# , dirichlet_tags="boundary")
U = TransientTrialFESpace(V0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

Γ_wall = BoundaryTriangulation(model, tags=[3, 4])
dΓ_wall = Measure(Γ_wall, 2)

Γ_ends = BoundaryTriangulation(model, tags=[1, 2])
dΓ_ends = Measure(Γ_ends, 2)

# TODO: replace w pyhysical
c = 1.0
rho = 1.0
Z0 = rho * c


Z_wall = 100.0 * Z0



res_vol(t, u, v) = ∫( ∂tt(u)*v + c^2 * (∇(u)⋅∇(v)) )dΩ
jac_vol(t, u, du, v) = ∫( c^2 * (∇(du)⋅∇(v)) )dΩ
jac_tt_vol(t, u, dutt, v) = ∫( dutt*v )dΩ

res_wall(t, u, v) = ∫( (1/Z_wall) * ∂t(u) * v )dΓ_wall
jac_t_wall(t, u, dut, v) = ∫( (1/Z_wall) * dut * v )dΓ_wall

res_ends(t, u, v) = ∫( (1/Z0) * ∂t(u) * v )dΓ_ends
jac_t_ends(t, u, dut, v) = ∫( (1/Z0) * dut * v )dΓ_ends

res(t, u, v) = res_vol(t,u,v) + res_wall(t,u,v) + res_ends(t,u,v)
jac(t, u, du, v) = jac_vol(t,u,du,v)
jac_tt(t, u, dutt, v) = jac_tt_vol(t,u,dutt,v)
jac_t(t, u, dut, v) = jac_t_wall(t,u,dut,v) + jac_t_ends(t,u,dut,v)
function u0_func(x)
    r2 = (x[1])^2  + (x[2]-0.5)^2
    return exp(-100*r2)
end
function v0_func(x)
    return 0.0
end
u0 = interpolate_everywhere(u0_func, U(0.0))
v0 = interpolate_everywhere(v0_func, U(0.0)) 
# res(t, u, v) = ∫( ∂tt(u)*v + c^2 * (∇(u)⋅∇(v)) )dΩ
# jac(t, u, du, v) = ∫( c^2 * (∇(du)⋅∇(v)) )dΩ
# jac_t(t, u, dut, v) = ∫( 0.0*dut*v )dΩ
# jac_tt(t, u, dutt, v) = ∫( dutt*v )dΩ
op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)
t0 = 0.0
t1 = 4.0
dt = 0.01
nonlinear_solver = NLSolver(show_trace=true, method=:newton) 
ode_solver = Newmark(nonlinear_solver, dt, 0.5, 0.25)
sol_t = solve(ode_solver, op, t0, t1, (u0, v0))
mkpath("wave_simulation/results")
createpvd("wave_simulation/results") do pvd
    pvd[0] = createvtk(Ω, "wave_simulation/results/wave_0" * ".vtu", cellfields=["u" => u0])
    for (tn, uh) in sol_t
        println("Solving at time $tn")
        pvd[tn] = createvtk(Ω, "wave_simulation/results/wave_$tn" * ".vtu", cellfields=["u"=>uh])
    end
end
