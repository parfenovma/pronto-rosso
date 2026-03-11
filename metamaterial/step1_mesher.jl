using Gmsh: gmsh
using LinearAlgebra

# === НАСТРОЙКИ ГЕНЕРАЦИИ ===
# Массив глубин синусоиды (0.0 - это плоский контрольный прямоугольник!)
AMPLITUDES = [0.0, 1.0, 2.0, 2.5] 
MESH_DIR = "1_meshes"

# Создаем папку, если ее нет
mkpath(MESH_DIR)

# Универсальная функция математики контура
function generate_crystal_points(A)
    B = 0.4; CC = 4.0; yy = 7.0; xx = 17.0 - 2.0*B; N_pts = 200
    C = A * CC / 100.0
    pts_bottom = Tuple{Float64, Float64}[]
    push!(pts_bottom, (0.0, 0.0))
    for i in 0:N_pts
        x = B + (4.0 * i * xx) / (5.0 * N_pts)
        angle_deg = (4.0 * 180.0 * i) / (N_pts + 1.0)
        if i <= N_pts/4 || i >= 3*N_pts/4
            y = (A / 2.0) * (1.0 - cosd(angle_deg))
        else
            y = (A / 2.0 - C / 2.0) * (1.0 - cosd(angle_deg)) + C
        end
        push!(pts_bottom, (x, y))
    end
    push!(pts_bottom, (xx + 2.0*B, 0.0)) 
    
    pts_top = Tuple{Float64, Float64}[]
    for p in pts_bottom
        push!(pts_top, (17.0 - p[1], yy - p[2]))
    end
    return vcat(pts_bottom, pts_top)
end

function build_mesh(A, filename)
    gmsh.clear() # Очищаем ядро для новой геометрии
    gmsh.model.add("PhonCrystal_A_$(A)")
    
    if A == 0.0
        # ОПТИМИЗАЦИЯ: Плоский референсный образец строим 1 строчкой
        surf_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, 17.0*1e-3, 7.0*1e-3)
        gmsh.model.occ.synchronize()
        lines_dimtags = gmsh.model.getEntities(1)
        line_tags = [tag for (dim, tag) in lines_dimtags]
    else
        # Сложный кристалл
        points_mm = generate_crystal_points(A)
        points = [(x * 1e-3, y * 1e-3) for (x, y) in points_mm]
        
        point_tags = Int[]
        prev_pt = (-100.0, -100.0)
        for (x, y) in points
            if norm([x - prev_pt[1], y - prev_pt[2]]) > 1e-6
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                push!(point_tags, tag)
                prev_pt = (x, y)
            end
        end
        
        line_tags = Int[]
        for i in 1:length(point_tags)
            p1 = point_tags[i]
            p2 = point_tags[i == length(point_tags) ? 1 : i+1] 
            push!(line_tags, gmsh.model.occ.addLine(p1, p2))
        end
        
        loop_tag = gmsh.model.occ.addCurveLoop(line_tags)
        surf_tag = gmsh.model.occ.addPlaneSurface([loop_tag])
        gmsh.model.occ.synchronize()
    end
    
    # Разметка
    s_lines, m_lines, f_lines = Int[], Int[], Int[]
    for tag in line_tags
        c_mass = gmsh.model.occ.getCenterOfMass(1, tag)
        x_c = c_mass[1]
        
        if x_c < 1e-5
            push!(s_lines, tag)   
        elseif x_c > (17.0 * 1e-3) - 1e-5
            push!(m_lines, tag)   
        else
            push!(f_lines, tag)   
        end
    end
    
    gmsh.model.addPhysicalGroup(1, s_lines, 101); gmsh.model.setPhysicalName(1, 101, "Source")
    gmsh.model.addPhysicalGroup(1, m_lines, 102); gmsh.model.setPhysicalName(1, 102, "Microphone")
    gmsh.model.addPhysicalGroup(1, f_lines, 103); gmsh.model.setPhysicalName(1, 103, "FreeSurface")
    gmsh.model.addPhysicalGroup(2, [surf_tag], 201); gmsh.model.setPhysicalName(2, 201, "Domain")
    
    # Сетка
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", vcat(s_lines, m_lines, f_lines))
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.15 * 1e-3)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.60 * 1e-3) 
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.2 * 1e-3)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 2.0 * 1e-3)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    
    gmsh.model.mesh.generate(2)
    gmsh.write(filename)
    println("  [+] Сетка сохранена: $filename")
end

println("=== ЗАПУСК ФАБРИКИ СЕТОК ===")
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0) # Отключаем спам в консоль

for A in AMPLITUDES
    filename = joinpath(MESH_DIR, "mesh_A_$(A).msh")
    build_mesh(A, filename)
end

gmsh.finalize()
println("=== ГЕНЕРАЦИЯ ЗАВЕРШЕНА ===")
