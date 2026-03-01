using Gridap
using Gridap.ODEs

# 1. СЕТКА
L = 3.0
H = 1.0
n = 100
model = CartesianDiscreteModel((0, L, 0, H), (3*n, n))

# 2. ПРОСТРАНСТВА ДЛЯ РЕШЕНИЯ (ВЕКТОРНЫЕ)
# Ищем вектор перемещений: u = (ux, uy)
reffe_vec = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
V0 = TestFESpace(model, reffe_vec, conformity=:H1) 
U = TransientTrialFESpace(V0)

# 2.1 ПРОСТРАНСТВО ДЛЯ МАТЕРИАЛОВ (СКАЛЯРНОЕ)
# Создаем пространство для скалярных полей (плотность, Ламе)
reffe_scal = ReferenceFE(lagrangian, Float64, 1)
V_scalar = FESpace(model, reffe_scal)

# Геометрия трубы
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

# Триангуляция и меры
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# 3. ПАРАМЕТРЫ МАТЕРИАЛА 
function rho_x(x)
    return is_inside_tube(x) ? 1.0 : 3.0
end

function lambda_x(x)
    return is_inside_tube(x) ? 5.0 : 15.0
end

function mu_x(x)
    return is_inside_tube(x) ? 4.0 : 12.0
end

function gamma_x(x)
    return is_inside_tube(x) ? 0.1 : 3.0
end

# ИНТЕРПОЛИРУЕМ СКАЛЯРЫ В СКАЛЯРНОЕ ПРОСТРАНСТВО V_scalar
rho_field = interpolate_everywhere(rho_x, V_scalar)
lambda_field = interpolate_everywhere(lambda_x, V_scalar)
mu_field = interpolate_everywhere(mu_x, V_scalar)
gamma_field = interpolate_everywhere(gamma_x, V_scalar)

# 4. ЗАКОН ГУКА
# one(ε) автоматически создает единичный тензор того же размера, что и ε
function σ(ε, λ, μ)
    return λ * tr(ε) * one(ε) + 2.0 * μ * ε
end

# 5. СЛАБАЯ ФОРМА
res(t, u, v) = ∫( 
    rho_field * ∂tt(u) ⋅ v +      
    gamma_field * rho_field * ∂t(u) ⋅ v + 
    σ∘(ε(u), lambda_field, mu_field) ⊙ ε(v)
)dΩ

jac(t, u, du, v) = ∫( σ∘(ε(du), lambda_field, mu_field) ⊙ ε(v) )dΩ

jac_t(t, u, dut, v) = ∫( gamma_field * rho_field * dut ⋅ v )dΩ

jac_tt(t, u, dutt, v) = ∫( rho_field * dutt ⋅ v )dΩ

# 6. НАЧАЛЬНЫЕ УСЛОВИЯ (Векторные)
function u0_func(x)
    r2 = (x[1])^2  + (x[2]-0.5)^2
    amp = exp(-100*r2)
    return VectorValue(amp, 0.0) # Импульс по X
end

function v0_func(x)
    return VectorValue(0.0, 0.0) # Нулевая начальная скорость
end

# ИНТЕРПОЛИРУЕМ ВЕКТОРЫ В ВЕКТОРНОЕ ПРОСТРАНСТВО U(0.0)
u0 = interpolate_everywhere(u0_func, U(0.0))
v0 = interpolate_everywhere(v0_func, U(0.0)) 

# 7. РЕШАТЕЛЬ И ЦИКЛ ПО ВРЕМЕНИ
op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)

t0 = 0.0
t1 = 4.0
dt = 0.02

nonlinear_solver = NLSolver(show_trace=true, method=:newton) 
ode_solver = Newmark(nonlinear_solver, dt, 0.5, 0.25)

sol_t = solve(ode_solver, op, t0, t1, (u0, v0))

# 8. ПАПКИ И СОХРАНЕНИЕ
out_dir = "elastic_wave_simulation/results"
mkpath(out_dir)

createpvd(out_dir) do pvd
    pvd[0] = createvtk(Ω, joinpath(out_dir, "wave_0.vtu"), 
                       cellfields=["u" => u0, "rho" => rho_field, "mu" => mu_field])
    for (tn, uh) in sol_t
        println("Solving at time $tn")
        pvd[tn] = createvtk(Ω, joinpath(out_dir, "wave_$(round(tn, digits=4)).vtu"), 
                            cellfields=["u"=>uh, "rho"=>rho_field, "mu"=>mu_field])
    end
end
println("Simulation finished successfully!")
