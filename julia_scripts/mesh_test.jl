using Gridap
using GridapGmsh
using Gmsh
using Gridap.ODEs
using Plots
using FFTW

# ==========================================
# 1. EXPERIMENT PARAMETERS
# ==========================================
freq = 220000.0       # 220 kHz
P_amplitude = 1e6     # АМПЛИТУДА ДАВЛЕНИЯ ПЭП (1 МПа)

# ==========================================
# 3. GRIDAP SETUP & PHYSICS
# ==========================================
println("Loading mesh...")
model = GmshDiscreteModel("crystal_mesh_2d.msh") 

rho_solid = 1150.0       
cp_solid = 2340.0        
cs_solid = 1170.0        
mu_solid = rho_solid * cs_solid^2 
lam_solid = rho_solid * cp_solid^2 - 2*mu_solid 
gamma_solid = 50000.0

# ==========================================
# 4. BOUNDARY CONDITIONS (PZT SOURCE)
# ==========================================
# Генератор радиоимпульса с окном Ханна
function pzt_signal(t)
    n_cycles = 4.0               
    duration = n_cycles / freq  
    
    if t < duration
        envelope = 0.5 * (1.0 - cos(2.0 * pi * t / duration))
        carrier = sin(2.0 * pi * freq * t)
        return envelope * carrier 
    else
        return 0.0
    end
end

reffe_vec = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
# Убрали dirichlet_tags! Край свободен, двигается под давлением
V0 = TestFESpace(model, reffe_vec, conformity=:H1, dirichlet_tags=["Microphone"]) 
u_zero(x,t) = VectorValue(0.0, 0.0)
u_zero(t::Real) = x -> u_zero(x,t) 
U = TransientTrialFESpace(V0, [u_zero])

# ==========================================
# 5. WEAK FORM & INTEGRATION MEASURES
# ==========================================
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# Микрофон (правый край)
Γ_out = BoundaryTriangulation(model, tags=["Microphone"])
dΓ_out = Measure(Γ_out, degree)
n_out = VectorValue(1.0, 0.0)

# Источник PZT (левый край)
Γ_in = BoundaryTriangulation(model, tags=["Source"])
dΓ_in = Measure(Γ_in, degree)
n_in = VectorValue(-1.0, 0.0) # Вектор нормали смотрит влево (-X)

σ(ε) = lam_solid * tr(ε) * one(ε) + 2.0 * mu_solid * ε
Z_p = rho_solid * cp_solid 

# res с учетом граничного условия Неймана (давления)
res(t, u, v) = ∫( rho_solid * ∂tt(u) ⋅ v + σ∘(ε(u)) ⊙ ε(v) )dΩ - 
               ∫( (P_amplitude * pzt_signal(t)) * (n_in ⋅ v) )dΓ_in

jac(t, u, du, v) = ∫( σ∘(ε(du)) ⊙ ε(v) )dΩ
jac_t(t, u, dut, v) = ∫( 0.0 * dut ⋅ v )dΩ 
jac_tt(t, u, dutt, v) = ∫( rho_solid * dutt ⋅ v )dΩ

# ==========================================
# 6. SOLVER INITIALIZATION
# ==========================================
t0 = 0.0
t1 = 100.0e-6 
dt = (1.0 / freq) / 30.0  

println("\nStarting simulation: Freq = $(freq/1000) kHz")

op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)

init_disp(x) = VectorValue(0.0, 0.0)
init_vel(x)  = VectorValue(0.0, 0.0)

U_at_t0 = U(0.0)
uh0 = interpolate_everywhere(init_disp, U_at_t0)
vh0 = interpolate_everywhere(init_vel, U_at_t0)

nonlinear_solver = NLSolver(show_trace=false, method=:newton) 
ode_solver = Newmark(nonlinear_solver, dt, 0.5, 0.25)
sol_t = solve(ode_solver, op, t0, t1, (uh0, vh0))

# ==========================================
# 7. MAIN LOOP: SOLVING & SENSOR LOGGING
# ==========================================
out_dir = "results_gmsh"
mkpath(out_dir)

time_history = Float64[]
signal_history_out_MPa = Float64[] # Храним давление в МПа

# Вычисляем длину границы микрофона (чтобы найти среднее давление)
L_mic = sum( ∫( 1.0 )dΓ_out )

createpvd(out_dir) do pvd
    step = 0
    save_every = 3 
    
    for (tn, uh) in sol_t
        # Считаем акустическое давление на микрофоне P = - σ_xx
        # σ∘(ε(uh)) - тензор напряжений. Умножаем на нормаль n_out дважды, чтобы получить скаляр давления.
        int_pressure = sum( ∫( - n_out ⋅ (σ∘(ε(uh)) ⋅ n_out) )dΓ_out )
        
        # Среднее давление на микрофоне переводим в Мегапаскали
        pressure_MPa = (int_pressure / L_mic) / 1e6
        
        push!(time_history, tn)
        push!(signal_history_out_MPa, pressure_MPa) 
        
        if step % save_every == 0
            pvd[tn] = createvtk(Ω, joinpath(out_dir, "wave_$(step).vtu"), cellfields=["u"=>uh])
        end
        step += 1
    end
end

# ==========================================
# 8. POST-PROCESSING (Plotting A-Scans)
# ==========================================
# Переводим время в микросекунды только для графиков
t_us = time_history .* 1e6

# Генерируем идеальный входной сигнал (Давление в Мегапаскалях)
in_signal_history = (P_amplitude / 1e6) .* pzt_signal.(time_history)

# График 1: Сигнал излучателя
p_i = scatter(t_us, in_signal_history, 
           title="Input Signal (PZT Pressure)", 
           ylabel="Pressure (MPa)",
           linewidth=2, color=:red, legend=false, grid=true)

# График 2: Сигнал на микрофоне
p_o = scatter(t_us, signal_history_out_MPa, 
           title="Output Signal (Microphone)", 
           xlabel="Time (us)", 
           ylabel="Pressure (MPa)",
           linewidth=2, color=:blue, legend=false, grid=true)

# Собираем оба графика один над другим
p_combined = plot(p_i, p_o, layout=(2, 1), size=(800, 600))

# Сохраняем картинки
savefig(p_i, "planar_mic_input.png")     # Отдельно вход
savefig(p_o, "planar_mic_signal.png")    # Отдельно выход
savefig(p_combined, "signals_combined.png") # Красивый совмещенный!

display(p_combined)

# ==========================================
# 9. SPECTRUM ANALYSIS (FFT)
# ==========================================
println("Calculating FFT...")

# 1. Считаем частоту дискретизации
dt_sim = time_history[2] - time_history[1] # реальный шаг по времени
fs = 1.0 / dt_sim                          # частота сэмплирования (Hz)
N = length(time_history)

# 2. Выполняем Быстрое Преобразование Фурье (FFT)
F_in = fft(in_signal_history)
F_out = fft(signal_history_out_MPa)

# 3. Получаем вектор частот (оставляем только положительную половину)
freqs = fftfreq(N, fs)
half_N = N ÷ 2
freqs_kHz = freqs[1:half_N] ./ 1000.0 # переводим в кГц для удобства

# 4. Считаем магнитуды (амплитуды) спектра
mag_in = abs.(F_in[1:half_N]) ./ N
mag_out = abs.(F_out[1:half_N]) ./ N

# 5. Рисуем графики спектра!
# Ограничим ось X до 800 кГц, чтобы смотреть только на полезную зону
p_fft_in = plot(freqs_kHz, mag_in, 
             title="Input Spectrum (What we sent)", 
             ylabel="Amplitude", 
             color=:red, linewidth=2, grid=true, legend=false,
             xlims=(0, 800))

p_fft_out = plot(freqs_kHz, mag_out, 
             title="Output Spectrum (What arrived)", 
             xlabel="Frequency (kHz)", 
             ylabel="Amplitude", 
             color=:blue, linewidth=2, grid=true, legend=false,
             xlims=(0, 800))

# Объединяем спектры
p_fft_combined = plot(p_fft_in, p_fft_out, layout=(2, 1), size=(800, 600))
savefig(p_fft_combined, "fft_spectrum.png")
display(p_fft_combined)