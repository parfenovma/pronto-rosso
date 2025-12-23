using Gridap
using Gridap.ODEs

# 1. СЕТКА (Покрывает всё пространство)
L = 3.0
H = 1.0
n = 100
model = CartesianDiscreteModel((0,L,0,H), (3*n, n)) 

# Обычные пространства
reffe = ReferenceFE(lagrangian, Float64, 1)
V0 = TestFESpace(model, reffe, conformity=:H1) 
# Можно добавить Dirichlet по краям большой коробки, если хотите
# V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags="boundary")
U = TransientTrialFESpace(V0)

function is_inside_tube(x)
    xi, yi = x[1], x[2]
    
    amp = 0.1
    freq = 2 * pi * 2.0

    offset = amp * sin(freq * xi)
    thickness = 0.2
    
    y_center = 0.5
    y_top = y_center + thickness + offset
    y_bot = y_center - thickness - offset
    
    return (yi > y_bot) && (yi < y_top)
end

degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

Γ_wall = BoundaryTriangulation(model, tags=[3, 4])
dΓ_wall = Measure(Γ_wall, 2)

Γ_ends = BoundaryTriangulation(model, tags=[1, 2])
dΓ_ends = Measure(Γ_ends, 2)

# TODO: replace w pyhysical
c_in = 3.0  # Внутри
c_out = 1.0 # Снаружи


function c_x(x)
    if is_inside_tube(x)
        return c_in
    else
        return c_out
    end
end


gamma_in = 0.1
gamma_out = 3.0
function gamma_x(x)
    if is_inside_tube(x)
        return gamma_in
    else
        return gamma_out
    end
end

rho_in = 1.0
rho_out = 3.0
function rho_x(x)
    if is_inside_tube(x)
        return rho_in
    else
        return rho_out
    end
end

c_field = interpolate_everywhere(c_x, U(0.0))
gamma_field = interpolate_everywhere(gamma_x, U(0.0))
rho_field = interpolate_everywhere(rho_x, U(0.0))

inv_bulk = 1.0 / (rho_field * c_field * c_field) # = 1 / (rho c^2)
inv_rho  = 1.0 / rho_field                       # = 1 / rho


# MASS TERM (Инерция): (1 / rho c^2) * u_tt
# DAMPING TERM (Потери): gamma * u_t 
# STIFFNESS TERM (Жесткость): (1 / rho) * grad(u) * grad(v)

res(t, u, v) = ∫( 
    inv_bulk * ∂tt(u) * v +      
    inv_bulk * gamma_field * ∂t(u) * v +
    inv_rho * (∇(u)⋅∇(v))
)dΩ

# В Якобиане тоже используем переменное c
jac(t, u, du, v) = ∫( inv_rho * (∇(du)⋅∇(v)) )dΩ
jac_tt(t, u, dutt, v) = ∫( inv_bulk * dutt * v )dΩ
# res(t, u, v) = ∫( ∂tt(u)*v + gamma_field*∂t(u)*v + (c_field*c_field) * (∇(u)⋅∇(v)) )dΩ
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
jac_t(t, u, dut, v) = ∫( 0.0*dut*v )dΩ
# jac_tt(t, u, dutt, v) = ∫( dutt*v )dΩ
op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)
t0 = 0.0
t1 = 4.0
dt = 0.02
nonlinear_solver = NLSolver(show_trace=true, method=:newton) 
ode_solver = Newmark(nonlinear_solver, dt, 0.5, 0.25)
sol_t = solve(ode_solver, op, t0, t1, (u0, v0))
mkpath("wave_simulation/results")
createpvd("wave_simulation/results") do pvd
    pvd[0] = createvtk(Ω, "wave_simulation/results/wave_0" * ".vtu", cellfields=["u" => u0, "c_map"=>c_field])
    for (tn, uh) in sol_t
        println("Solving at time $tn")
        pvd[tn] = createvtk(Ω, "wave_simulation/results/wave_$tn" * ".vtu", cellfields=["u"=>uh, "c_map"=>c_field])
    end
end
