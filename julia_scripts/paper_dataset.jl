using Gridap
using Gridap.ODEs
using DataFrames
using CSV
using Plots

# ==========================================
# 1. НАСТРОЙКИ СКАНА (SWEEP ПАРАМЕТРОВ)
# ==========================================
# Частоты из статьи: 220 кГц и 830 кГц
frequencies = [100000.0, 150000.0, 220000.0, 500000.0, 830000.0]

# Изгибы от 0.0 до 3.6 мм с шагом 0.4 мм (как в статье)
# (В метрах: от 0.0 до 0.0036 с шагом 0.0004)
amps_bend = collect(0.0 : 0.0004 : 0.0044)

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
    amp_mm = round(amp_bend * 1000, digits=2)
    println("\n==================================================")
    println("▶ Запуск симуляции: Изгиб = $amp_mm мм (Частота: $(freq/1000) кГц)")
    println("==================================================")

    # Уникальное имя для сетки
    mesh_file = "temp_waveguide_$(amp_mm)mm.msh"
    L = 0.025; h_thick = 0.007; y_c_base = 0.0075     

    # --- 1. GMSH ---
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) 
    gmsh.clear()
    gmsh.model.add("Waveguide")

    get_yc(x) = (0.005 < x < 0.020) ? (y_c_base + amp_bend * sin(2.0*pi*(x-0.005)/0.015)) : y_c_base

    N_pts = 80
    xs = range(0, L, length=N_pts)
    top_pts, bot_pts = Int32[], Int32[]

    for x in xs
        push!(bot_pts, gmsh.model.geo.addPoint(x, get_yc(x) - h_thick/2, 0))
        push!(top_pts, gmsh.model.geo.addPoint(x, get_yc(x) + h_thick/2, 0))
    end

    c_loop = gmsh.model.geo.addCurveLoop([
        gmsh.model.geo.addSpline(bot_pts),
        gmsh.model.geo.addLine(bot_pts[end], top_pts[end]),
        gmsh.model.geo.addSpline(reverse(top_pts)),
        gmsh.model.geo.addLine(top_pts[1], bot_pts[1])
    ])
    
    surf = gmsh.model.geo.addPlaneSurface([c_loop])
    gmsh.model.geo.synchronize()

    # ВАЖНО: Физические группы
    gmsh.model.addPhysicalGroup(1, [4], -1, "left_pzt") # Левая линия была 4-й
    gmsh.model.addPhysicalGroup(1, [2], -2, "right_mic") # Правая линия была 2-й
    gmsh.model.addPhysicalGroup(2, [surf], -3, "solid_domain")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.0004)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0001)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_file)
    gmsh.finalize()

    # --- 2. GRIDAP SETUP ---
    model = GmshDiscreteModel(mesh_file) 
    rho_solid = 1150.0; cp_solid = 2340.0; cs_solid = 1170.0        
    mu_solid = rho_solid * cs_solid^2 
    lam_solid = rho_solid * cp_solid^2 - 2*mu_solid 
    gamma_solid = 50000.0
    Z_p = rho_solid * cp_solid  

    u_D(x, t) = t < (4.0 / freq) ? VectorValue(1e-6 * 0.5 * (1 - cos(2 * pi * freq * t / 4.0)) * sin(2 * pi * freq * t), 0.0) : VectorValue(0.0, 0.0)
    u_D(t::Real) = x -> u_D(x, t)

    V0 = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2, Float64}, 1), conformity=:H1, dirichlet_tags=["left_pzt"]) 
    U = TransientTrialFESpace(V0, [u_D])

    dΩ = Measure(Triangulation(model), 2)
    dΓ_out = Measure(BoundaryTriangulation(model, tags=["right_mic"]), 2)

    σ(ε) = lam_solid * tr(ε) * one(ε) + 2.0 * mu_solid * ε
    res(t, u, v) = ∫( rho_solid * ∂tt(u) ⋅ v + gamma_solid * rho_solid * ∂t(u) ⋅ v + σ∘(ε(u)) ⊙ ε(v) )dΩ + ∫( Z_p * ∂t(u) ⋅ v )dΓ_out
    jac(t, u, du, v) = ∫( σ∘(ε(du)) ⊙ ε(v) )dΩ
    jac_t(t, u, dut, v) = ∫( 0.0 * dut ⋅ v )dΩ + ∫( Z_p * dut ⋅ v )dΓ_out
    jac_tt(t, u, dutt, v) = ∫( rho_solid * dutt ⋅ v )dΩ

    # --- 3. РЕШЕНИЕ ---
    dt = (1.0 / freq) / 25.0  
    t1 = 60.0e-6 # 60 мкс достаточно для 220 кГц
    
    op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)
    uh0 = interpolate_everywhere(x->VectorValue(0.0, 0.0), U(0.0))
    sol_t = solve(Newmark(NLSolver(show_trace=false, method=:newton), dt, 0.5, 0.25), op, 0.0, t1, (uh0, uh0))

    # # 2.7 СБОР ДАННЫХ (ВИРТУАЛЬНЫЙ ДАТЧИК)
    # max_amplitude = 0.0
    arrival_time = 0.0
    
    # # Мы не пишем VTU файлы, мы сканируем поле!
    # for (tn, uh) in sol_t
    #     # Вычисляем перемещение в точке датчика
    #     u_val = norm(uh(sensor_point))
        
    #     # Ищем максимум огибающей (пик пакета)
    #     if u_val > max_amplitude
    #         max_amplitude = u_val
    #         arrival_time = tn
    #     end
    # end

    # --- 4. СБОР ДАННЫХ ---
    num_sensors = 40
    y_coords = range(0.004 + 0.0001, 0.011 - 0.0001, length=num_sensors)
    sensor_points = [Point(L - 0.0001, y) for y in y_coords]
    max_amplitude = 0.0
    

    time_vals = Float64[]
    beam_profile = zeros(Float64, num_sensors, 0)

    for (tn, uh) in sol_t
        push!(time_vals, tn * 1e6)
        current_snapshot = [uh(p)[1] * 1e9 for p in sensor_points] 
        beam_profile = hcat(beam_profile, current_snapshot)
        if sum(current_snapshot) > max_amplitude
            max_amplitude = sum(current_snapshot)
            arrival_time = tn
        end
    end

    rm(mesh_file, force=true)
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
