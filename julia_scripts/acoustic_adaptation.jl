using Gridap
using Gridap.ODEs

n = 100
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

reffe = ReferenceFE(lagrangian, Float64, 1)

V0 = TestFESpace(model, reffe, conformity=:H1)# , dirichlet_tags="boundary")
U = TransientTrialFESpace(V0)

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
c = 1.0
function u0_func(x)
    r2 = (x[1])^2 #  + (x[2]-0.5)^2
    return exp(-100*r2)
end
function v0_func(x)
    return 0.0
end
u0 = interpolate_everywhere(u0_func, U(0.0))
v0 = interpolate_everywhere(v0_func, U(0.0)) 
res(t, u, v) = ∫( ∂tt(u)*v + c^2 * (∇(u)⋅∇(v)) )dΩ
jac(t, u, du, v) = ∫( c^2 * (∇(du)⋅∇(v)) )dΩ
jac_t(t, u, dut, v) = ∫( 0.0*dut*v )dΩ
jac_tt(t, u, dutt, v) = ∫( dutt*v )dΩ
op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)
t0 = 0.0
t1 = 2.0
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
