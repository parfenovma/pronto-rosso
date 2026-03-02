using Gridap
using GridapGmsh
using Gmsh
using Gridap.ODEs
using Plots

# ==========================================
# 1. EXPERIMENT PARAMETERS
# ==========================================
freq = 220000.0       # Frequency in Hz
amp_bend = 0.003      # Corrugation amplitude in meters (3 mm)

# Waveguide geometry
L = 0.025             
h_thick = 0.007       
y_c_base = 0.0075     

mesh_file = "waveguide_current.msh" 

# ==========================================
# 2. GMSH MESH GENERATOR
# ==========================================
function generate_waveguide_mesh(filename, amplitude)
    println("Generating Gmsh mesh for Amplitude = $(amplitude*1000) mm...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) 
    gmsh.clear()
    gmsh.model.add("Waveguide")

    get_yc(x) = (0.005 < x < 0.020) ? (y_c_base + amplitude * sin(2.0*pi*(x-0.005)/0.015)) : y_c_base

    N_pts = 80
    xs = range(0, L, length=N_pts)
    top_pts, bot_pts = Int32[], Int32[]

    for x in xs
        yc = get_yc(x)
        push!(bot_pts, gmsh.model.geo.addPoint(x, yc - h_thick/2, 0))
        push!(top_pts, gmsh.model.geo.addPoint(x, yc + h_thick/2, 0))
    end

    line_bottom = gmsh.model.geo.addSpline(bot_pts)
    line_right  = gmsh.model.geo.addLine(bot_pts[end], top_pts[end])
    line_top    = gmsh.model.geo.addSpline(reverse(top_pts))
    line_left   = gmsh.model.geo.addLine(top_pts[1], bot_pts[1])

    curve_loop = gmsh.model.geo.addCurveLoop([line_bottom, line_right, line_top, line_left])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    gmsh.model.geo.synchronize()

    # Physical Groups (Tags are mapped to Gridap automatically)
    gmsh.model.addPhysicalGroup(1, [line_left], -1, "left_pzt")
    gmsh.model.addPhysicalGroup(1, [line_right], -2, "right_mic") # <-- Planar Microphone Setup!
    gmsh.model.addPhysicalGroup(2, [surface], -3, "solid_domain")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.0004)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0001)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 40)

    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    gmsh.finalize()
end

generate_waveguide_mesh(mesh_file, amp_bend)

# ==========================================
# 3. GRIDAP SETUP & PHYSICS
# ==========================================
model = GmshDiscreteModel(mesh_file) 

rho_solid = 1150.0       
cp_solid = 2340.0        
cs_solid = 1170.0        
mu_solid = rho_solid * cs_solid^2 
lam_solid = rho_solid * cp_solid^2 - 2*mu_solid 

# ==========================================
# 4. BOUNDARY CONDITIONS (PZT SOURCE)
# ==========================================
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
V0 = TestFESpace(model, reffe_vec, conformity=:H1, dirichlet_tags=["left_pzt"]) 
U = TransientTrialFESpace(V0, [u_D])

# ==========================================
# 5. WEAK FORM & INTEGRATION MEASURES
# ==========================================
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# PLANAR MICROPHONE MEASURE
Γ_out = BoundaryTriangulation(model, tags=["right_mic"])
dΓ_out = Measure(Γ_out, degree)
n_Γ = VectorValue(1.0, 0.0) 

σ(ε) = lam_solid * tr(ε) * one(ε) + 2.0 * mu_solid * ε

Z_p = rho_solid * cp_solid  

res(t, u, v) = ∫( rho_solid * ∂tt(u) ⋅ v + σ∘(ε(u)) ⊙ ε(v) )dΩ + 
               ∫( Z_p * ∂t(u) ⋅ v )dΓ_out

jac(t, u, du, v) = ∫( σ∘(ε(du)) ⊙ ε(v) )dΩ

jac_t(t, u, dut, v) = ∫( 0.0 * dut ⋅ v )dΩ + 
                      ∫( Z_p * dut ⋅ v )dΓ_out

jac_tt(t, u, dutt, v) = ∫( rho_solid * dutt ⋅ v )dΩ

# ==========================================
# 6. SOLVER INITIALIZATION
# ==========================================
t0 = 0.0
t1 = 60.0e-6 
dt = (1.0 / freq) / 15.0  

println("\nStarting simulation: Freq = $(freq/1000) kHz, Bend = $(amp_bend*1000) mm")

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
signal_history = Float64[]

createpvd(out_dir) do pvd
    step = 0
    save_every = 3 
    
    for (tn, uh) in sol_t
        # CORE UPDATE: Integrate displacement over the entire output plane!
        integrated_signal = sum( ∫( uh ⋅ n_Γ )dΓ_out )
        
        push!(time_history, tn * 1e6)
        push!(signal_history, integrated_signal)
        
        if step % save_every == 0
            pvd[tn] = createvtk(Ω, joinpath(out_dir, "wave_$(step).vtu"), cellfields=["u"=>uh])
        end
        step += 1
    end
end

println("Simulation finished. Files saved to $out_dir.")

# ==========================================
# 8. POST-PROCESSING (Planar Microhone Plot)
# ==========================================
p = plot(time_history, signal_history, 
         title="Planar Microphone Output ($(freq/1000) kHz)", 
         xlabel="Time (us)", 
         ylabel="Integrated Displacement (Area, m²)",
         linewidth=2, legend=false, grid=true, color=:blue)

savefig(p, "planar_mic_signal.png")
display(p)

# Cleanup Gmsh cache
rm(mesh_file, force=true)
