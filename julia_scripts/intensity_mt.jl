using Gridap
using Gridap.ODEs
using Plots

# ==========================================
# 1. ОСНОВНОЙ РЕШАТЕЛЬ (Возвращает сигнал и макс. амплитуду)
# ==========================================
function get_signal(freq, amp_bend)
    L = 0.025; H = 0.015  
    # Немного уплотнили сетку для избавления от ступенек
    model = CartesianDiscreteModel((0, L, 0, H), (250, 150))
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

    dt = (1.0 / freq) / 15.0  
    t1 = 30.0e-6 
    op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)

    U_0 = U(0.0)
    uh0 = interpolate_everywhere(x->VectorValue(0.0, 0.0), U_0)
    vh0 = interpolate_everywhere(x->VectorValue(0.0, 0.0), U_0)

    sol_t = solve(Newmark(NLSolver(show_trace=false, method=:newton), dt, 0.5, 0.25), op, 0.0, t1, (uh0, vh0))

    u_arr = Float64[]
    max_amp = 0.0
    sensor = Point(0.024, 0.0075)

    for (tn, uh) in sol_t
        val = uh(sensor)[1]
        push!(u_arr, val)
        if abs(val) > max_amp
            max_amp = abs(val)
        end
    end
    
    return u_arr, dt, max_amp
end

# ==========================================
# 2. АЛГОРИТМ КРОСС-КОРРЕЛЯЦИИ
# ==========================================
function get_phase_delay(u_ref, u_sig, dt)
    n = length(u_ref)
    max_corr = -Inf
    best_shift = 0
    correlations = zeros(150) # Ищем сдвиг глубже
    
    for shift in 1:149
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
    
    if best_shift > 1 && best_shift < 149
        y1 = correlations[best_shift - 1]
        y2 = correlations[best_shift]
        y3 = correlations[best_shift + 1]
        fractional_shift = (y1 - y3) / (2 * (y1 - 2*y2 + y3))
        exact_shift = best_shift + fractional_shift
    else
        exact_shift = best_shift
    end
    
    return exact_shift * dt
end

# ==========================================
# 3. ОСНОВНАЯ МНОГОПОТОЧНАЯ ЛОГИКА
# ==========================================
frequencies = [100000.0, 220000.0, 500000.0, 830000.0]
amps_bend = collect(0.0 : 0.0004 : 0.0036)

# Подготовим два поля для графиков
p1 = plot(title="Phase Delay vs Bending Depth", ylabel="Added Delay (us)", legend=:topleft, grid=true)
p2 = plot(title="Signal Transmission", xlabel="Bending Depth (mm)", ylabel="Amplitude (%)", legend=:bottomleft, grid=true)

println("Starting execution with $(Threads.nthreads()) threads 🚀")

for freq in frequencies
    freq_khz = round(Int, freq / 1000)
    println("\n--- Processing: $freq_khz kHz ---")
    println("Calculating Reference (0.0 mm)...")
    
    u_ref, dt, max_amp_ref = get_signal(freq, 0.0)
    
    delays = zeros(length(amps_bend))
    transmissions = zeros(length(amps_bend))
    
    Threads.@threads for i in 1:length(amps_bend)
        amp = amps_bend[i]
        
        u_sig, _, max_amp_sig = get_signal(freq, amp)
        
        # 1. Считаем задержку
        delay_s = get_phase_delay(u_ref, u_sig, dt)
        delays[i] = delay_s * 1e6
        
        # 2. Считаем проценты прошедшей энергии!
        transmissions[i] = (max_amp_sig / max_amp_ref) * 100.0
    end
    
    # Корректируем базовый ноль
    corrected_delays = delays .- delays[1]
    
    # Рисуем линии для этой частоты на обоих графиках
    plot!(p1, amps_bend .* 1000, corrected_delays, marker=:circle, linewidth=2, label="$freq_khz kHz")
    plot!(p2, amps_bend .* 1000, transmissions, marker=:square, linewidth=2, label="$freq_khz kHz")
end

# Склеиваем 2 графика в один кадр
final_plot = plot(p1, p2, layout=(2,1), size=(800, 800))

# ==========================================
# 4. СОХРАНЕНИЕ
# ==========================================
savefig(final_plot, "metamaterial_analysis.png")
display(final_plot)
println("\nDONE! Your masterpiece is saved as 'metamaterial_analysis.png'")
