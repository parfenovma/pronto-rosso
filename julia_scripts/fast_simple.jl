using Gridap, LinearAlgebra
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Plots  # Для визуализации

# Параметры модели
Lx = 1.0  # Размер области по x
Ly = 1.0  # Размер области по y
c = 1.0   # Скорость звука
T = 1.0   # Общее время моделирования
dt = 0.005  # Шаг по времени (должен быть маленьким для стабильности)
n = 100   # Число элементов по каждой оси (сетка n x n)

# Создание геометрии и сетки
domain = (0, Lx, 0, Ly)
partition = (n, n)
model = CartesianDiscreteModel(domain, partition)

# Определение конечных элементов (лагранжевы полиномы степени 1)
reffe = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags="boundary")

# Начальные условия: гауссов импульс в центре
σ = 0.05
p0(x) = exp(-((x[1]-Lx/2)^2 + (x[2]-Ly/2)^2)/(2σ^2))
U0 = interpolate_everywhere(p0, V)  # Начальное давление
V0 = interpolate_everywhere(0.0, V)  # Начальная скорость (∂p/∂t = 0)

# Слабая форма волнового уравнения (для FEM)
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
m(u,v) = ∫( v⋅u )dΩ

# Матрицы массы и жёсткости
M = assemble_matrix(m, V, V)
K = assemble_matrix(a, V, V)

# Временная дискретизация (Newmark метод, β=0.25, γ=0.5 для неявной схемы)
α = 1/(4dt^2)
β = 1/(2dt)
γ = c^2

# Инициализация
P = copy(get_free_dof_values(U0))  # Текущее давление
V = copy(get_free_dof_values(V0))  # Текущая скорость
A = similar(P)  # Ускорение

# Функция для обновления (неявная схема Newmark)
function update!(P, V, A, t)
    # Правая часть: -K*P + внешние силы (здесь 0)
    rhs = -γ * (K * P)
    # Решение (M + α*dt^2 * γ K) A = rhs + M*(α*P + β*V + ...), но упрощённо
    # Для простоты используем явную схему центральных разностей
    # A = M \ (rhs - K*P)  # Но для явной: лучше инвертировать M
end

# Для стабильности используем явную центральную разностную схему
invM = lu(M)  # Факторизация массы для быстрого решения

# Визуализация
anim = @animate for t in 0:dt:T
    # Вычисление ускорения A = invM * (-K * P)
    mul!(A, K, P)
    A .= -A
    ldiv!(invM, A)  # A = invM * (-K*P)
    
    # Обновление скорости и позиции (явная схема)
    V .+= dt * A
    P .+= dt * V
    
    # Интерполяция для визуализации
    ph = FEFunction(V, P)
    
    # Визуализация
    p = plot(Ω, ph, clim=(-1,1), title="t = $t")
    display(p)
end

# Сохранение анимации
gif(anim, "acoustic_wave_2d.gif", fps=15)
