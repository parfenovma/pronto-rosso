using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Visualization
using Gridap.ODEs

# Параметры модели
domain = (0, 1, 0, 1)  # Квадратный домен [0,1] x [0,1]
partition = (50, 50)   # Сетка 50x50
model = CartesianDiscreteModel(domain, partition)

order = 1  # Порядок элементов (линейные)
reffe = ReferenceFE(lagrangian, Float64, order)
V = TestFESpace(model, reffe, dirichlet_tags="boundary")  # Тестовое пространство с нулевыми границами

g(t) = x -> 0.0  # Нулевые Dirichlet-условия как функция
U = TransientTrialFESpace(V, g)  # Пробное пространство

# Многофилдовые пространства для (u, v=∂t u)
Y = MultiFieldFESpace([V, V])
X = TransientMultiFieldFESpace([U, U])

# Интеграция
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# Параметры волны
c = 1.0  # Скорость волны

# Слабые формы для affine оператора
m(t, (du, dv), (p, q)) = ∫(du*p + dv*q) * dΩ
a(t, (u, v), (p, q)) = ∫(-v*p + c^2 * ∇(q)⋅∇(u)) * dΩ
b(t, (p, q)) = ∫(0.0) * dΩ  # Нет источника

# Оператор (affine для линейной проблемы)
op = TransientAffineFEOperator(m, a, b, X, Y)

# Солвер для времени (ThetaMethod, второй порядок)
ls = LUSolver()  # Линейный солвер
Δt = 0.005       # Шаг по времени (маленький для стабильности CFL)
θ = 0.5          # Для второго порядка
odesolver = ThetaMethod(ls, Δt, θ)

# Начальные условия
u0(x) = exp(-50 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2))  # Гауссова "вспышка" в центре
v0(x) = 0.0  # Начальная скорость = 0

t0 = 0.0
tF = 1.0  # Конечное время

yh0 = interpolate_everywhere((u0, v0), X(t0))

# Решение
sol_t = solve(odesolver, op, t0, tF, yh0)

# Сохранение результатов для анимации (VTK + PVD)
mkpath("wave_results")
pvd = createpvd("wave_results/wave")

local step = 0  # Чтобы избежать предупреждения о scope
for (tn, yhn) in sol_t
    (uhn, vhn) = yhn
    if step % 5 == 0  # Сохраняем каждый 5-й шаг для экономии
        pvd[tn] = createvtk(Ω, "wave_results/wave_$step.vtu", cellfields=["u" => uhn])
    end
    step += 1
end

# Сохраняем PVD-файл
savepvd(pvd)