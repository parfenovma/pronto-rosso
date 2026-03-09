using Gmsh: gmsh
using LinearAlgebra

# --- 1. ПАРАМЕТРЫ ФОНОННОГО КРИСТАЛЛА (из кода коллеги) ---
const yy = 7.0       # Высота (размер по y)
const zz = 7.0       # Толщина для 3D печати (размер по z)
const B = 0.4        # Прямые торцы
const CC = 5.0       # Смещение амплитуды
const N_pts = 200    # Дискретизация сплайна
const xx = 17.0 - 2*B  # Длина волновой части

export_3d_stl = false # Флаг: сохранять ли 3D модель для коллеги?
mesh_2d_msh   = true # Флаг: генерировать 2D сетку для тебя?

# --- 2. МАТЕМАТИКА КОНТУРА (Магия вместо "вырезания кусочков") ---
# Функция, которая вычисляет координаты точек по формулам коллеги
function generate_crystal_points(A)
    C = A * CC / 100.0
    pts_bottom = Tuple{Float64, Float64}[]
    
    # 2.1 Плоский участок слева-внизу
    push!(pts_bottom, (0.0, 0.0))
    # push!(pts_bottom, (B, 0.0)) # (убираем, чтобы не дублировать i=0)
    
    # 2.2 Синусоидальный профиль нижней части
    for i in 0:N_pts
        x = B + (4.0 * i * xx) / (5.0 * N_pts)
        angle_deg = (4.0 * 180.0 * i) / (N_pts + 1.0)
        
        # Точь-в-точь по формулам из SCAD
        if i <= N_pts/4 || i >= 3*N_pts/4
            y = (A / 2.0) * (1.0 - cosd(angle_deg))
        else
            y = (A / 2.0 - C / 2.0) * (1.0 - cosd(angle_deg)) + C
        end
        push!(pts_bottom, (x, y))
    end
    
    # 2.3 Плоский участок справа-внизу (добиваем до 17.0)
    # Коллега в SCAD ставил квадрат в конце, мы просто ведем прямую линию:
    push!(pts_bottom, (xx + 2B, 0.0)) # xx + 2B = 17.0
    
    # 2.4 Верхний профиль
    # В SCAD это решалось через rotate(180). В математике это просто:
    # x_top = 17 - x_bottom, y_top = 7 - y_bottom.
    # Добавляем в обратном порядке от X=17 обратно к X=0, чтобы замкнуть контур
    pts_top = Tuple{Float64, Float64}[]
    for p in pts_bottom
        push!(pts_top, (17.0 - p[1], yy - p[2]))
    end
    
    # Склеиваем всё в один единый замкнутый массив точек по часовой стрелке!
    return vcat(pts_bottom, pts_top)
end

# === ЗАПУСК GMSH И ГЕНЕРАЦИЯ ГЕОМЕТРИИ === #

println("==> Инициализация геометрического ядра (OpenCASCADE)...")
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("PhononicCrystal")

# Создаем точки по математической функции (Вызываем с глубиной волны A = 2.0)
points_mm = generate_crystal_points(2.0)
points = [(x * 1e-3, y * 1e-3) for (x, y) in points_mm]

# 1. Загружаем точки в Gmsh, фильтруя микро-дубликаты (защита от ошибок)
point_tags = Int[]
global previous_pt = (-100.0, -100.0)
for (x, y) in points
    if norm([x - previous_pt[1], y - previous_pt[2]]) > 1e-6
        tag = gmsh.model.occ.addPoint(x, y, 0.0)
        push!(point_tags, tag)
        global previous_pt = (x, y)
    end
end

# 2. Соединяем точки линиями, образуя кольцо
line_tags = Int[]
for i in 1:length(point_tags)
    p1 = point_tags[i]
    p2 = point_tags[i == length(point_tags) ? 1 : i+1] # Замыкаем последнюю на первую
    tag = gmsh.model.occ.addLine(p1, p2)
    push!(line_tags, tag)
end

# 3. Делаем единую правильную плоскость без всяких "булевых костылей"
println("==> Построение 2D поверхности...")
loop_tag = gmsh.model.occ.addCurveLoop(line_tags)
surf_tag = gmsh.model.occ.addPlaneSurface([loop_tag])
gmsh.model.occ.synchronize()


# === 3D ЭКСПОРТ ДЛЯ КОЛЛЕГИ (Магия) === #
if export_3d_stl
    println("==> 🎁 Экспорт 3D модели для принтера...")
    # Копируем 2D плоскость и выдавливаем на толщину zz = 7.0
    # [(2, surf_tag)] = массив объектов размерности 2 и с тегом surf_tag
    extrude_out = gmsh.model.occ.extrude([(2, surf_tag)], 0.0, 0.0, zz)
    gmsh.model.occ.synchronize()
    
    # Для STL нужна просто сетка по поверхностям 3D тела (Mesh 2D)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5) 
    gmsh.model.mesh.generate(2) # Генерирует треугольники
    gmsh.write("phonon_crystal_3d_print.stl")
    println("\tСохранен файл: phonon_crystal_3d_print.stl")
    
    # Возвращаем Gmsh в плоский "2D режим" для твоей симуляции
    gmsh.clear()
    gmsh.model.add("PhononicCrystal2D")
    
    # Пересобираем плоскую поверхность за миллисекунду
    loop_tag = gmsh.model.occ.addCurveLoop(line_tags)
    surf_tag = gmsh.model.occ.addPlaneSurface([loop_tag])
    gmsh.model.occ.synchronize()
end


# === РАЗМЕТКА PHYSICAL GROUPS ДЛЯ ТВОЕГО GRIDAP === #
if mesh_2d_msh
    println("==> 🔬 Настройка 2D симуляции (Разметка)...")
    
    s_lines, m_lines, f_lines = Int[], Int[], Int[]
    
    # Так как мы сами строили линии, найти их легко по Центру Масс
    for tag in line_tags
        c_mass = gmsh.model.occ.getCenterOfMass(1, tag)
        x_c = c_mass[1]
        
        # Допуск 1e-5 метров (это 0.01 мм)
        if x_c < 1e-5
            push!(s_lines, tag)   # Нашлись микро-линии на X = 0.000 м
        elseif x_c > (17.0 * 1e-3) - 1e-5
            push!(m_lines, tag)   # Нашлись микро-линии на X = 0.017 м (Микрофон)
        else
            push!(f_lines, tag)   # Всё остальное
        end
    end
    
    # Присваиваем имена физическим группам
    gmsh.model.addPhysicalGroup(1, s_lines, 101)
    gmsh.model.setPhysicalName(1, 101, "Source")
    
    gmsh.model.addPhysicalGroup(1, m_lines, 102)
    gmsh.model.setPhysicalName(1, 102, "Microphone")
    
    gmsh.model.addPhysicalGroup(1, f_lines, 103)
    gmsh.model.setPhysicalName(1, 103, "FreeSurface")
    
    gmsh.model.addPhysicalGroup(2, [surf_tag], 201)
    gmsh.model.setPhysicalName(2, 201, "Domain")
    
    # === НАСТРОЙКА УПЛОТНЕНИЯ СЕТКИ ===
    # Делаем сетку мелкой (0.1) возле рельефа и крупной (0.6) внутри!
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
    
    # === ГЕНЕРАЦИЯ 2D СЕТКИ ===
    println("==> Построение 2D сетки для Gridap...")
    gmsh.model.mesh.generate(2)
    gmsh.write("crystal_mesh_2d.msh")
    println("\tСохранен файл: crystal_mesh_2d.msh")
end

# Если хочешь посмотреть и покрутить красивый результат ручками - раскомментируй:
# gmsh.fltk.run()

gmsh.finalize()
println("==> Готово. Можно запускать акустический решатель!")
