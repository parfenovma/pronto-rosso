using Gridap
using Gridap.ODEs
using DataFrames
using CSV
using Plots

# ==========================================
# 1. НАСТРОЙКИ СКАНА (SWEEP ПАРАМЕТРОВ)
# ==========================================
# Частоты из статьи: 220 кГц и 830 кГц
frequencies = [220000.0, 830000.0]

# Изгибы от 0.0 до 3.6 мм с шагом 0.4 мм (как в статье)
# (В метрах: от 0.0 до 0.0036 с шагом 0.0004)
amps_bend = collect(0.0 : 0.0004 : 0.0036)

# Координата "виртуального микрофона" на выходе из трубы (x=24мм, y=7.5мм)
sensor_point = Point(0.024, 0.0075)

# Таблица для хранения результатов
results_df = DataFrame(Frequency_Hz = Float64[], AmpBend_mm = Float64[], DelayTime_us = Float64[])

# ==========================================
# 2. ОСНОВНАЯ ФУНКЦИЯ СИМУЛЯЦИИ
# ==========================================
# Эта функция принимает частоту и амплитуду изгиба, проводит расчет 
# и возвращает время прихода максимального сигнала (в секундах).
function run_simulation(freq, amp_bend)
    
    # 2.1 Сетка
    L = 0.025  
    H = 0.015  
    model = CartesianDiscreteModel((0, L, 0, H), (200, 120))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "left_pzt", [1, 3, 7])

    # 2.2 Геометрия 
    function is_inside_waveguide(x)
        xi, yi = x[1], x[2]
        h = 0.007 
        y_c = 0.0075 
        if 0.005 < xi < 0.020
            phase = 2.0 * pi * (xi - 0.005) / 0.015
            y_c += amp_bend * sin(phase) 
        end
        return (yi > y_c - h/2) && (yi < y_c + h/2)
    end

    # 2.3 Материалы (Воздух теперь "Мягкий", без затухания)
    rho_solid = 1150.0       
    cp_solid = 2340.0        
    cs_solid = 1170.0        
    mu_solid = rho_solid * cs_solid^2 
    lam_solid = rho_solid * cp_solid^2 - 2*mu_solid 

    rho_air = 1.0            
    lam_air = 1.0            
    mu_air = 1.0             
    gamma_air = 0.0          

    function rho_x(x) return is_inside_waveguide(x) ? rho_solid : rho_air end
    function lam_x(x) return is_inside_waveguide(x) ? lam_solid : lam_air end
    function mu_x(x)  return is_inside_waveguide(x) ? mu_solid  : mu_air  end
    function gam_x(x) return 0.0 end # Затухание убрали для чистоты эксперимента

    reffe_scal = ReferenceFE(lagrangian, Float64, 1)
    V_scalar = FESpace(model, reffe_scal)

    rho_f = interpolate_everywhere(rho_x, V_scalar)
    lam_f = interpolate_everywhere(lam_x, V_scalar)
    mu_f  = interpolate_everywhere(mu_x, V_scalar)
    gam_f = interpolate_everywhere(gam_x, V_scalar)

    # 2.4 Генератор
    function pzt_signal(t)
        n_periods = 4.0
        duration = n_periods / freq
        if t < duration
            amp = 1e-6 * 0.5 * (1 - cos(2 * pi * freq * t / n_periods))
            return amp * sin(2 * pi * freq * t)
        else
            return 0.0
        end
    end

    u_D(x, t) = VectorValue(pzt_signal(t), 0.0)
    u_D(t::Real) = x -> u_D(x, t)

    reffe_vec = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    V0 = TestFESpace(model, reffe_vec, conformity=:H1, labels=labels, dirichlet_tags=["left_pzt"]) 
    U = TransientTrialFESpace(V0, [u_D])

    # 2.5 Уравнения
    degree = 2
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    function σ(ε, λ, μ) return λ * tr(ε) * one(ε) + 2.0 * μ * ε end

    res(t, u, v) = ∫( rho_f * ∂tt(u) ⋅ v + gam_f * rho_f * ∂t(u) ⋅ v + σ∘(ε(u), lam_f, mu_f) ⊙ ε(v) )dΩ
    jac(t, u, du, v) = ∫( σ∘(ε(du), lam_f, mu_f) ⊙ ε(v) )dΩ
    jac_t(t, u, dut, v) = ∫( gam_f * rho_f * dut ⋅ v )dΩ
    jac_tt(t, u, dutt, v) = ∫( rho_f * dutt ⋅ v )dΩ

    # 2.6 Решатель
    t0 = 0.0
    t1 = 35.0e-6 
    dt = (1.0 / freq) / 10.0  

    op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)

    # Уникальные имена для начальных условий, чтобы избежать конфликтов
    init_d(x) = VectorValue(0.0, 0.0)
    init_v(x)  = VectorValue(0.0, 0.0)
    U_at_t0 = U(0.0)
    uh0 = interpolate_everywhere(init_d, U_at_t0)
    vh0 = interpolate_everywhere(init_v, U_at_t0)

    nonlinear_solver = NLSolver(show_trace=false, method=:newton) 
    ode_solver = Newmark(nonlinear_solver, dt, 0.5, 0.25)
    sol_t = solve(ode_solver, op, t0, t1, (uh0, vh0))

    # 2.7 СБОР ДАННЫХ (ВИРТУАЛЬНЫЙ ДАТЧИК)
    max_amplitude = 0.0
    arrival_time = 0.0
    
    # Мы не пишем VTU файлы, мы сканируем поле!
    for (tn, uh) in sol_t
        # Вычисляем перемещение в точке датчика
        u_val = norm(uh(sensor_point))
        
        # Ищем максимум огибающей (пик пакета)
        if u_val > max_amplitude
            max_amplitude = u_val
            arrival_time = tn
        end
    end
    
    return arrival_time
end

# ==========================================
# 3. ЗАПУСК ЦИКЛОВ (СБОР ДАТАСЕТА)
# ==========================================
println("Starting parameter sweep... This will take a while!")

for freq in frequencies
    println("\n--- Testing Frequency: $(freq / 1000) kHz ---")
    
    for amp in amps_bend
        print("  Running Amp = $(amp * 1000) mm... ")
        
        # Запускаем симуляцию
        t_arr = run_simulation(freq, amp)
        t_arr_us = t_arr * 1e6 # Переводим в микросекунды
        
        println("Delay: $(round(t_arr_us, digits=2)) us")
        
        # Добавляем строку в датасет
        push!(results_df, (freq, amp * 1000, t_arr_us))
    end
end

# ==========================================
# 4. СОХРАНЕНИЕ И ОТРИСОВКА
# ==========================================
# Сохраняем в CSV
CSV.write("delay_dataset.csv", results_df)
println("\nDataset saved to 'delay_dataset.csv'!")
display(results_df)

# Строим красивый график прямо как в статье!
p = plot(title="Delay Time vs Bending Depth", 
         xlabel="Bending Depth (mm)", 
         ylabel="Delay Time (us)", 
         legend=:topleft, 
         grid=true)

for freq in frequencies
    df_f = filter(row -> row.Frequency_Hz == freq, results_df)
    plot!(p, df_f.AmpBend_mm, df_f.DelayTime_us, 
          marker=:circle, linewidth=2, label="$(freq/1000) kHz")
end

# Сохраняем график в картинку
savefig(p, "delay_plot.png")
println("Plot saved to 'delay_plot.png'.")
