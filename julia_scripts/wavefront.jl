using Gridap
using GridapGmsh
using Gmsh
using Gridap.ODEs
using Plots

# ==========================================
# 1. ГЛАВНАЯ ФУНКЦИЯ АНАЛИЗА ОДНОЙ ГЕОМЕТРИИ
# ==========================================
function generate_wavefront_profile(freq, amp_bend)
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
    dt = (1.0 / freq) / 15.0  
    t1 = 60.0e-6 # 60 мкс достаточно для 220 кГц
    
    op = TransientFEOperator(res, (jac, jac_t, jac_tt), U, V0)
    uh0 = interpolate_everywhere(x->VectorValue(0.0, 0.0), U(0.0))
    sol_t = solve(Newmark(NLSolver(show_trace=false, method=:newton), dt, 0.5, 0.25), op, 0.0, t1, (uh0, uh0))

    # --- 4. СБОР ДАННЫХ ---
    num_sensors = 40
    y_coords = range(0.004 + 0.0001, 0.011 - 0.0001, length=num_sensors)
    sensor_points = [Point(L - 0.0001, y) for y in y_coords]

    time_vals = Float64[]
    beam_profile = zeros(Float64, num_sensors, 0)

    for (tn, uh) in sol_t
        push!(time_vals, tn * 1e6)
        current_snapshot = [uh(p)[1] * 1e9 for p in sensor_points] 
        beam_profile = hcat(beam_profile, current_snapshot)
    end

    # --- 5. ГРАФИКА И СОХРАНЕНИЕ ---
    max_val = maximum(abs.(beam_profile))
    
    p = heatmap(time_vals, y_coords .* 1000, beam_profile,
                c = :RdBu, clims = (-max_val, max_val),
                title = "Wavefront: Bend $amp_mm mm (220 kHz)",
                xlabel = "Time (us)", ylabel = "Y Position (mm)",
                right_margin = 5Plots.mm)

    filename = "profile_bend_$(amp_mm)mm.png"
    savefig(p, filename)
    display(p)

    # Удаляем временную сетку
    rm(mesh_file, force=true)
    println("✔ Готово! Сохранено как $filename")
end

# ==========================================
# 2. ЦИКЛ ПО АМПЛИТУДАМ (САМА ИТЕРАЦИЯ)
# ==========================================
freq_test = 190000.0  # 220 kHz

# От прямой трубы (0.0 мм) до глубокой (3.6 мм) из статьи
amplitudes_to_test = [0.0, 0.0012, 0.0024, 0.0036, 0.0048]

for amp in amplitudes_to_test
    # Вызываем нашу функцию для каждого изгиба
    generate_wavefront_profile(freq_test, amp)
end

println("\n🎉 Процесс завершен! Все 4 картинки сохранены в рабочей папке.")
