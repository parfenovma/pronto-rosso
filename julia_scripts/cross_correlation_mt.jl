using Gridap
using Gridap.ODEs
using Plots

# ==========================================
# 1. ФУНКЦИЯ ПОЛУЧЕНИЯ СИГНАЛА С МИКРОФОНА
# ==========================================
function get_signal(freq, amp_bend)
    L = 0.025; H = 0.015  
    model = CartesianDiscreteModel((0, L, 0, H), (200, 120))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "left_pzt", [1, 3, 7])

    function is_inside_waveguide(x)
        xi, yi = x[1], x[2]
        h = 0.007; y_c = 0.0075 
        if 0.005 < xi < 0.020
            y_c += amp_bend * sin(2.0 * pi * (xi - 0.005) / 0.015) 
        end
        return (yi > y_c - h/2) && (yi < y_c + h/2)
    end

    rho_solid = 1150.0; cp_solid = 2340.0; cs_solid = 1170.0        
    mu_solid = rho_solid * cs_solid^2 
    lam_solid = rho_solid * cp_solid^2 - 2*mu_solid 
    rho_air = 1.0; lam_air = 1.0; mu_air = 1.0; gamma_air = 0.0          

    rho_x(x) = is_inside_waveguide(x) ? rho_solid : rho_air
    lam_x(x) = is_inside_waveguide(x) ? lam_solid : lam_air
    mu_x(x)  = is_inside_waveguide(x) ? mu_solid  : mu_air
    gam_x(x) = 0.0

    V_scalar = FESpace(model, ReferenceFE(lagrangian, Float64, 1))
    rho_f = interpolate_everywhere(rho_x, V_scalar)
    lam_f = interpolate_everywhere(lam_x, V_scalar)
    mu_f  = interpolate_everywhere(mu_x, V_scalar)
    gam_f = interpolate_everywhere(gam_x, V_scalar)

    function pzt_signal(t)
        if t < (4.0 / freq)
            return 1e-6 * 0.5 * (1 - cos(2 * pi * freq * t / 4.0)) * sin(2 * pi * freq * t)
        end
        return 0.0
    end

    u_D(x, t) = VectorValue(pzt_signal(t), 0.0)
    u_D(t::Real) = x -> u_D(x, t)

    V0 = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2, Float64}, 1), conformity=:H1, labels=labels, dirichlet_tags=["left_pzt"]) 
    U = TransientTrialFESpace(V0, [u_D])

    dΩ = Measure(Triangulation(model), 2)
    σ(ε, λ, μ) = λ * tr(ε) * one(ε) + 2.0 * μ * ε

    res(t, u, v) = ∫( rho_f * ∂tt(u) ⋅ v + gam_f * rho_f * ∂t(u) ⋅ v + σ∘(ε(u), lam_f, mu_f) ⊙ ε(v) )dΩ
    jac(t, u, du, v) = ∫( σ∘(ε(du), lam_f, mu_f) ⊙ ε(v) )dΩ
    jac_t(t, u, dut, v) = ∫( 0.0 * dut ⋅ v )dΩ
    jac_tt(t, u, dutt, v) = ∫( rho_f * dutt ⋅ v )dΩ

    dtime = (1.0 / freq) / 20.0  
    t1 = 30.0e-6 
    op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)

    U_0 = U(0.0)
    uh0 = interpolate_everywhere(x->VectorValue(0.0, 0.0), U_0)
    vh0 = interpolate_everywhere(x->VectorValue(0.0, 0.0), U_0)

    sol_t = solve(Newmark(NLSolver(show_trace=false, method=:newton), dtime, 0.5, 0.25), op, 0.0, t1, (uh0, vh0))

    u_arr = Float64[]
    sensor = Point(0.024, 0.0075)

    for (tn, uh) in sol_t
        push!(u_arr, uh(sensor)[1]) 
    end
    
    return u_arr, dtime
end

# ==========================================
# 2. АЛГОРИТМ ФАЗОВОЙ ВЗАИМОКОРРЕЛЯЦИИ
# ==========================================
function get_phase_delay(u_ref, u_sig, dtime)
    n = length(u_ref)
    max_corr = -Inf
    best_shift = 0
    # Ищем задержку в пределах разумного (100 шагов по времени)
    correlations = zeros(100) 
    
    for shift in 1:99
        corr = 0.0
        for i in 1:(n - shift)
            corr += u_ref[i] * u_sig[i + shift]
        end
        correlations[shift] = corr
        
        if corr > max_corr
            max_corr = corr
            best_shift = shift
        end
    end
    
    # Параболическая интерполяция для суб-шаговой точности
    if best_shift > 1 && best_shift < 99
        y1 = correlations[best_shift - 1]
        y2 = correlations[best_shift]
        y3 = correlations[best_shift + 1]
        
        fractional_shift = (y1 - y3) / (2 * (y1 - 2*y2 + y3))
        exact_shift = best_shift + fractional_shift
    else
        exact_shift = best_shift
    end
    
    return exact_shift * dtime
end

# ==========================================
# 3. ОСНОВНАЯ МНОГОПОТОЧНАЯ ЛОГИКА
# ==========================================
frequencies = [220000.0, 830000.0]
amps_bend = collect(0.0 : 0.0004 : 0.0036)

# Подготовим график заранее
p = plot(title="Phase Delay vs Bending Depth (Cross-Corr)", 
         xlabel="Bending Depth (mm)", 
         ylabel="Added Phase Delay (us)", 
         legend=:topleft, 
         grid=true)

# Проверяем количество потоков
println("Starting execution with $(Threads.nthreads()) threads...")

for freq in frequencies
    println("\n--- Testing Frequency: $(freq / 1000) kHz ---")
    println("Getting REFERENCE signal (0.0 mm)...")
    
    # Эталон считаем один раз для данной частоты
    u_ref, dtime = get_signal(freq, 0.0)
    
    # Резервируем массив под задержки, чтобы избежать гонки потоков
    delays_for_freq = zeros(length(amps_bend))
    
    println("Starting parallel sweep for $(length(amps_bend)) geometries...")
    
    # МНОГОПОТОЧНЫЙ ЦИКЛ ЗДЕСЬ
    Threads.@threads for i in 1:length(amps_bend)
        amp = amps_bend[i]
        println("  [Thread $(Threads.threadid())] Processing: $(amp * 1000) mm...")
        
        u_sig, _ = get_signal(freq, amp)
        delay_s = get_phase_delay(u_ref, u_sig, dtime)
        
        delays_for_freq[i] = delay_s * 1e6 # в микросекундах
    end
    
    # Сдвигаем график, чтобы 0 мм изгиба всегда был равен 0 мкс задержки
    baseline_correction = delays_for_freq[1]
    corrected_delays = delays_for_freq .- baseline_correction
    
    # Добавляем линию на график
    plot!(p, amps_bend .* 1000, corrected_delays, 
          marker=:circle, linewidth=2, label="$(freq/1000) kHz")
end

# ==========================================
# 4. СОХРАНЕНИЕ 
# ==========================================
savefig(p, "perfect_phase_delay.png")
display(p)
println("\nComputation completed! Graphic saved as 'perfect_phase_delay.png'")
