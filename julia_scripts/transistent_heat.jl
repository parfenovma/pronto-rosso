using Gridap
domain = (-1, +1, -1, +1)
partition = (20, 20)
model = CartesianDiscreteModel(domain, partition)
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V0 = TestFESpace(model, reffe, dirichlet_tags="boundary")
g(t) = x -> exp(-2 * t) * sinpi(t * x[1]) * (x[2]^2 - 1)
Ug = TransientTrialFESpace(V0, g)
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
α(t) = x -> 1 + sin(t) * (x[1]^2 + x[2]^2) / 4
f(t) = x -> sin(t) * sinpi(x[1]) * sinpi(x[2])
m(t, dtu, v) = ∫(v * dtu)dΩ
a(t, u, v) = ∫(α(t) * ∇(v) ⋅ ∇(u))dΩ
l(t, v) = ∫(v * f(t))dΩ
op = TransientLinearFEOperator((a, m), l, Ug, V0)
op_opt = TransientLinearFEOperator((a, m), l, Ug, V0, constant_forms=(true, false))
ls = LUSolver()
Δt = 0.05
θ = 0.5
solver = ThetaMethod(ls, Δt, θ)
tableau = :SDIRK_2_2
solver_rk = RungeKutta(ls, ls, Δt, tableau)
t0, tF = 0.0, 10.0
uh0 = interpolate_everywhere(g(t0), Ug(t0))
uh = solve(solver, op, t0, tF, uh0)
mkpath("output_path/results")

createpvd("output_path/results") do pvd
  pvd[0] = createvtk(Ω, "output_path/results/results_0" * ".vtu", cellfields=["u" => uh0])
  for (tn, uhn) in uh
    pvd[tn] = createvtk(Ω, "output_path/results/results_$tn" * ".vtu", cellfields=["u" => uhn])
  end
end