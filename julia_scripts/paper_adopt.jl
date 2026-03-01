using Gridap
using Gridap.ODEs

# ==========================================
# 1. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА (МЕНЯТЬ ЗДЕСЬ)
# ==========================================
# Выбираем частоту (220 кГц - рабочая, 830 кГц - контрольная)
# freq = 830000.0 
freq = 830000.0  

# Глубина изгиба (Амплитуда гофры). В статье от 0 до 3.6 мм.
amp_bend = 0.003 # 3 мм

# ==========================================
# 2. ГЕОМЕТРИЯ И СЕТКА
# ==========================================
# Волновод 25 мм в длину и "коробка" 15 мм в высоту (м)
L = 0.025  
H = 0.015  

# Сетка: 150x90 ячеек (достаточно мелко для 830 кГц)
model = CartesianDiscreteModel((0, L, 0, H), (300, 180))

# Маркируем левую границу (для пьезодинамика)
labels = get_face_labeling(model)
# У CartesianDiscreteModel левая грань имеет теги 1, 3 и 7. 
add_tag_from_tags!(labels, "left_pzt", [1, 3, 7])

# ==========================================
# 3. ФУНКЦИЯ ФОРМЫ ВОЛНОВОДА
# ==========================================
function is_inside_waveguide(x)
    xi, yi = x[1], x[2]
    h = 0.007 # Тощина волновода - 7 мм по статье
    y_c = 0.0075 # Центр коробки по высоте
    
    # Синусоидальный изгиб начинается с 5 мм и заканчивается на 20 мм (длина 15 мм)
    if 0.005 < xi < 0.020
        phase = 2.0 * pi * (xi - 0.005) / 0.015
        y_c += amp_bend * sin(phase) 
    end
    
    return (yi > y_c - h/2) && (yi < y_c + h/2)
end

# ==========================================
# 4. ФИЗИЧЕСКИЕ СВОЙСТВА (Фотополимер vs Воздух)
# ==========================================
# Свойства пластика:
rho_solid = 1150.0       # кг/м^3
cp_solid = 2340.0        # м/с (скорость продольной волны)
cs_solid = 1170.0        # м/с (скорость поперечной волны)
mu_solid = rho_solid * cs_solid^2 # Модуль сдвига (~1.57 ГПа)
lam_solid = rho_solid * cp_solid^2 - 2*mu_solid # Параметр Ламе (~3.15 ГПа)

# Свойства воздуха (мягкий и легкий фиктивный домен):
# Свойства воздуха - делаем его сверхлегким и мягким, БЕЗ ЗАТУХАНИЯ!
rho_air = 1.0            
lam_air = 1.0            # Почти ноль
mu_air = 1.0             # Почти ноль
gamma_air = 0.0

function rho_x(x) return is_inside_waveguide(x) ? rho_solid : rho_air end
function lam_x(x) return is_inside_waveguide(x) ? lam_solid : lam_air end
function mu_x(x)  return is_inside_waveguide(x) ? mu_solid  : mu_air  end
function gam_x(x) return is_inside_waveguide(x) ? 0.0       : gamma_air end

# ==========================================
# 5. ПРОСТРАНСТВА И ИНТЕРПОЛЯЦИЯ МАТЕРИАЛОВ
# ==========================================
reffe_scal = ReferenceFE(lagrangian, Float64, 1)
V_scalar = FESpace(model, reffe_scal)

rho_f = interpolate_everywhere(rho_x, V_scalar)
lam_f = interpolate_everywhere(lam_x, V_scalar)
mu_f  = interpolate_everywhere(mu_x, V_scalar)
gam_f = interpolate_everywhere(gam_x, V_scalar)

# ==========================================
# 6. ПЬЕЗОДИНАМИК И ГРАНИЧНЫЕ УСЛОВИЯ
# ==========================================
# Генерируем "Тон-пакет" - 4 периода синусоиды, умноженные на "окошко" Хэннинга
function pzt_signal(t)
    n_periods = 4.0
    duration = n_periods / freq
    if t < duration
        # Амплитуда 1 микрометр (1e-6)
        amp = 1e-6 * 0.5 * (1 - cos(2 * pi * freq * t / n_periods))
        return amp * sin(2 * pi * freq * t)
    else
        return 0.0
    end
end

# Граничное условие смещает левый торец по оси X
u_D(x, t) = VectorValue(pzt_signal(t), 0.0)
u_D(t::Real) = x -> u_D(x, t)

# Векторное пространство со встроенным граничным условием
reffe_vec = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
V0 = TestFESpace(model, reffe_vec, conformity=:H1, labels=labels, dirichlet_tags=["left_pzt"]) 
U = TransientTrialFESpace(V0, [u_D])

# ==========================================
# 7. УРАВНЕНИЯ (Эластодинамика)
# ==========================================
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

function σ(ε, λ, μ)
    return λ * tr(ε) * one(ε) + 2.0 * μ * ε
end

res(t, u, v) = ∫( 
    rho_f * ∂tt(u) ⋅ v +      
    gam_f * rho_f * ∂t(u) ⋅ v + 
    σ∘(ε(u), lam_f, mu_f) ⊙ ε(v)
)dΩ

jac(t, u, du, v) = ∫( σ∘(ε(du), lam_f, mu_f) ⊙ ε(v) )dΩ
jac_t(t, u, dut, v) = ∫( gam_f * rho_f * dut ⋅ v )dΩ
jac_tt(t, u, dutt, v) = ∫( rho_f * dutt ⋅ v )dΩ

# ==========================================
# 8. МАСШТАБЫ ВРЕМЕНИ И РЕШАТЕЛЬ
# ==========================================
# Волна 2340 м/с пролетит 25 мм примерно за 10 микросекунд. Симулируем 20 микросекунд.
t0 = 0.0
t1 = 35.0e-6 
# Шаг по времени задаем так, чтобы хорошо разрешать частоту (~10 точек на период)
dt = (1.0 / freq) / 10.0  

println("Starting simulation for frequency = $(freq/1000) kHz")
println("Total time: $(t1 * 1e6) us, Timestep: $(dt * 1e6) us")

op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)

# Нулевые начальные условия везде (всё запускается от левой стенки)
init_disp(x) = VectorValue(0.0, 0.0)
init_vel(x)  = VectorValue(0.0, 0.0)

U_at_t0 = U(0.0)
uh0 = interpolate_everywhere(init_disp, U_at_t0)
vh0 = interpolate_everywhere(init_vel, U_at_t0)

nonlinear_solver = NLSolver(show_trace=false, method=:newton) 
ode_solver = Newmark(nonlinear_solver, dt, 0.5, 0.25)
sol_t = solve(ode_solver, op, t0, t1, (uh0, vh0))


# ==========================================
# 9. СОХРАНЕНИЕ 
# ==========================================
out_dir = "metamaterial_sim/freq_$(round(Int, freq/1000))"
mkpath(out_dir)

createpvd(out_dir) do pvd
    step = 0
    # Сохраняем каждый 3-й шаг чтобы не забить диск
    save_every = 3 
    
    for (tn, uh) in sol_t
        if step % save_every == 0
            time_us = round(tn * 1e6, digits=2)
            println("Time: $time_us us")
            pvd[tn] = createvtk(Ω, joinpath(out_dir, "wave_$(step).vtu"), 
                                cellfields=["u"=>uh, "rho"=>rho_f])
        end
        step += 1
    end
end
println("Simulation finished! Open files in $out_dir")
