using Gmsh

# ==========================================
# 1. ПАРАМЕТРЫ ГЕОМЕТРИИ
# ==========================================
L = 0.025        # Длина (м)
h = 0.007        # Толщина волновода (м)
y_c_base = 0.0075# Базовая высота центра (м)
amp_bend = 0.003 # Амплитуда гофры (м)

# Функция для вычисления Y координаты центра волновода
function get_y_center(x)
    if 0.005 < x < 0.020
        phase = 2.0 * pi * (x - 0.005) / 0.015
        return y_c_base + amp_bend * sin(phase)
    else
        return y_c_base
    end
end

# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ GMSH
# ==========================================
gmsh.initialize()
gmsh.clear()
gmsh.model.add("Waveguide")

# ==========================================
# 3. ПОСТРОЕНИЕ ГЕОМЕТРИИ
# ==========================================
# Создаем массивы точек для верхней и нижней границ.
# Берем 100 точек по длине X, чтобы сплайн идеально повторил синус.
N_pts = 100
xs = range(0, L, length=N_pts)

top_pts = Int32[]
bot_pts = Int32[]

# Создаем точки
for x in xs
    y_c = get_y_center(x)
    
    # Нижняя точка
    pt_b = gmsh.model.geo.addPoint(x, y_c - h/2, 0)
    push!(bot_pts, pt_b)
    
    # Верхняя точка
    pt_t = gmsh.model.geo.addPoint(x, y_c + h/2, 0)
    push!(top_pts, pt_t)
end

# Соединяем точки гладкими сплайнами и линиями
# Важно: линии должны образовывать замкнутый контур (петлю), 
# поэтому обходим по часовой (или против), например: Нижняя -> Правая -> Верхняя -> Левая

line_bottom = gmsh.model.geo.addSpline(bot_pts)
line_right  = gmsh.model.geo.addLine(bot_pts[end], top_pts[end])
# Для верхней линии инвертируем порядок точек, чтобы идти справа налево
line_top    = gmsh.model.geo.addSpline(reverse(top_pts)) 
line_left   = gmsh.model.geo.addLine(top_pts[1], bot_pts[1])

# Создаем замкнутый контур из линий
curve_loop = gmsh.model.geo.addCurveLoop([line_bottom, line_right, line_top, line_left])

# Создаем поверхность внутри контура
surface = gmsh.model.geo.addPlaneSurface([curve_loop])

gmsh.model.geo.synchronize()

# ==========================================
# 4. ФИЗИЧЕСКИЕ ГРУППЫ (Нужны для Gridap)
# ==========================================
# Маркируем левую границу для приложения пьезосигнала
gmsh.model.addPhysicalGroup(1, [line_left], -1, "left_pzt")

# Маркируем саму поверхность (2D домен)
gmsh.model.addPhysicalGroup(2, [surface], -1, "domain")


# ==========================================
# 5. НАСТРОЙКИ СЕТКИ (Уплотнение по кривизне)
# ==========================================
# Максимальный размер ячейки (на прямых участках)
# Зависит от длины волны: c=2340, f=830кГц -> lambda ~2.8 мм. Берем размер элементов < 0.3 мм
max_h = 0.0004 
# Минимальный размер ячейки (в местах сильного изгиба)
min_h = 0.0001 

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_h)

# ВКЛЮЧАЕМ уплотнение по кривизне
gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
# Сколько элементов должно приходиться на полный круг изгиба (2*pi)
# Чем больше число, тем сильнее уплотняется сетка на поворотах
gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 40)

# ==========================================
# 6. ГЕНЕРАЦИЯ И ЭКСПОРТ
# ==========================================
# Генерируем 2D сетку
gmsh.model.mesh.generate(2)

# Сохраняем в формат VTK для просмотра в ParaView
gmsh.write("waveguide_mesh.vtk")

# Сохраняем в формат MSH для последующего импорта в Gridap
gmsh.write("waveguide_mesh.msh")

println("Сетка успешно сгенерирована! Открой файл 'waveguide_mesh.vtk' в ParaView.")

gmsh.finalize()
