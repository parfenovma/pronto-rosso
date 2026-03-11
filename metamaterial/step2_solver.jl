using Gridap
using Gridap.ODEs
using GridapGmsh
using JLD2


const GMSH_LOCK = ReentrantLock()

# === ПАРАМЕТРЫ СВИПА (СЕТКА ЭКСПЕРИМЕНТОВ) ===
AMPLITUDES = [0.0, 1.0, 2.0]     # Должны совпадать с теми, для которых есть .msh
FREQUENCIES = [200e3, 220e3]     # Массив частот для прогонки
SAVE_VTK = false                 # ВАЖНО: Ставь false при массовом расчете, чтобы не забить SSD!

MESH_DIR = "1_meshes"
VTK_DIR = "2_vtks"
SIG_DIR = "3_signals"

mkpath(VTK_DIR)
mkpath(SIG_DIR)

# Функция одного эксперимента (Инкапсулирована для потокобезопасности)
function run_acoustic_simulation(A, freq)
    mesh_file = joinpath(MESH_DIR, "mesh_A_$(A).msh")
    if !isfile(mesh_file)
        println("  [!] Пропуск: Сетка $mesh_file не найдена!")
        return
    end

    println("  -> Старт: Амплитуда A=$A, Частота=$(freq/1000) кГц на потоке $(Threads.threadid())")

    # ЗАГРУЗКА И ФИЗИКА
    model = lock(GMSH_LOCK) do
        GmshDiscreteModel(mesh_file)
    end
    rho_solid = 1150.0; cp_solid = 2340.0; cs_solid = 1170.0        
    mu_solid = rho_solid * cs_solid^2; lam_solid = rho_solid * cp_solid^2 - 2*mu_solid 
    alpha_damp = 0.0; beta_damp = 2.0e-8 

    # Сигнал
    function pzt_signal(t)
        duration = 4.0 / freq
        t < duration ? 0.5 * (1.0 - cos(2.0 * pi * t / duration)) * sin(2.0 * pi * freq * t) : 0.0
    end

    # Пространства (Микрофон жесткий!)
    reffe_vec = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    V0 = TestFESpace(model, reffe_vec, conformity=:H1, dirichlet_tags=["Microphone"]) 
    U = TransientTrialFESpace(V0, [t -> (x -> VectorValue(0.0, 0.0))])

    # Интегралы
    degree = 2
    Ω = Triangulation(model); dΩ = Measure(Ω, degree)
    Γ_out = BoundaryTriangulation(model, tags=["Microphone"]); dΓ_out = Measure(Γ_out, degree)
    Γ_in = BoundaryTriangulation(model, tags=["Source"]); dΓ_in = Measure(Γ_in, degree)
    n_out = VectorValue(1.0, 0.0); n_in = VectorValue(-1.0, 0.0)

    σ(ε) = lam_solid * tr(ε) * one(ε) + 2.0 * mu_solid * ε
    P_amplitude = 1e6 

    res(t, u, v) = ∫( rho_solid * ∂tt(u) ⋅ v + alpha_damp * rho_solid * ∂t(u) ⋅ v + 
                      beta_damp * (σ∘(ε(∂t(u))) ⊙ ε(v)) + σ∘(ε(u)) ⊙ ε(v) )dΩ - 
                   ∫( (P_amplitude * pzt_signal(t)) * (n_in ⋅ v) )dΓ_in

    jac(t, u, du, v) = ∫( σ∘(ε(du)) ⊙ ε(v) )dΩ
    jac_t(t, u, dut, v) = ∫( alpha_damp * rho_solid * dut ⋅ v + beta_damp * (σ∘(ε(dut)) ⊙ ε(v)) )dΩ
    jac_tt(t, u, dutt, v) = ∫( rho_solid * dutt ⋅ v )dΩ

    # Решатель
    t1 = 60.0e-6 # 60 мкс (хватит для фиксации эха)
    dt = (1.0 / freq) / 30.0  
    op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)
    
    U_at_t0 = U(0.0)
    uh0 = interpolate_everywhere(x -> VectorValue(0.0, 0.0), U_at_t0)
    vh0 = interpolate_everywhere(x -> VectorValue(0.0, 0.0), U_at_t0)
    
    nonlinear_solver = NLSolver(show_trace=false, method=:newton) 
    ode_solver = Newmark(nonlinear_solver, dt, 0.5, 0.25)
    sol_t = solve(ode_solver, op, 0.0, t1, (uh0, vh0))

    # СБОР ДАННЫХ
    time_history = Float64[]
    signal_history_out_MPa = Float64[]
    L_mic = sum( ∫( 1.0 )dΓ_out )

    # Если включен экспорт анимации
    pvd = SAVE_VTK ? createpvd(joinpath(VTK_DIR, "anim_A_$(A)_F_$(freq/1000)")) : nothing
    step = 0
    
    for (tn, uh) in sol_t
        push!(time_history, tn)
        int_pressure = sum( ∫( - n_out ⋅ (σ∘(ε(uh)) ⋅ n_out) )dΓ_out )
        push!(signal_history_out_MPa, (int_pressure / L_mic) / 1e6)
        
        if SAVE_VTK && step % 3 == 0
            pvd[tn] = createvtk(Ω, joinpath(VTK_DIR, "anim_A_$(A)_F_$(freq/1000)_$(step).vtu"), cellfields=["u"=>uh])
        end
        step += 1
    end
    
    if SAVE_VTK; savepvd(pvd); end

    # Восстанавливаем входной сигнал для сохранения
    in_signal_history = (P_amplitude / 1e6) .* pzt_signal.(time_history)

    # === DUMP В БАЗУ ДАННЫХ (JLD2) ===
    save_path = joinpath(SIG_DIR, "data_A_$(A)_F_$(freq/1000).jld2")
    jldsave(save_path; 
        A = A, 
        freq = freq, 
        time = time_history, 
        signal_in = in_signal_history, 
        signal_out = signal_history_out_MPa
    )
    
    println("  [+] Готово: A=$A, Freq=$(freq/1000) кГц -> Сохранено в $save_path")
end

# === ГЛАВНЫЙ МНОГОПОТОЧНЫЙ ЦИКЛ ===
println("=== СТАРТ РАСЧЕТОВ (Потоков доступно: $(Threads.nthreads())) ===")

# Собираем все комбинации параметров в один плоский массив заданий
tasks = [(A, freq) for A in AMPLITUDES for freq in FREQUENCIES]

# Магический тег @threads раскидает задачи по процессорам Мака!
Threads.@threads for params in tasks
    A, freq = params
    run_acoustic_simulation(A, freq)
end

println("=== ВСЕ РАСЧЕТЫ ЗАВЕРШЕНЫ УСПЕШНО! ===")
