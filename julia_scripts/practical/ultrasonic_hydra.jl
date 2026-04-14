### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 84118270-3504-11f1-820f-733e877b1a89
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add("PlutoUI")
    Pkg.add("Plots")
    Pkg.add("SpecialFunctions")
    
    using PlutoUI
    using Plots
    using SpecialFunctions
end


# ╔═╡ 70ae7041-a8d1-4712-87db-8bdbdc9bf190
md"""
# Лабораторная работа
## Искажение формы и поглощение мощных ультразвуковых волн
**Данные эксперимента автоматизированы и загружены из таблицы Excel (`Акустика прак 4.xlsx`).**
"""


# ╔═╡ 4831a806-25b7-4252-bc9a-3657e81225dd
begin
	# 1. Данные для расчета эквивалентной длительности имп. (Скважности)
	A_max = 2.3
	A_raw = [0.5, 1, 1.3, 1.5, 1.9, 2.1, 2.2, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.2, 2.1, 1.9, 1.7, 1.6, 1.4, 1.1, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
	
	# 2. Данные эволюции 2-й гармоники
	# Расстояние x в см
	x_exp_cm = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
	# Амплитуда второй гармоники A2
	A2_exp = [48, 48, 60, 92, 112, 140, 188, 236, 272, 308, 332, 352, 352, 344, 332, 320, 312, 300, 288, 276, 272, 244, 232, 224]
end


# ╔═╡ 93e11d08-7abc-49c9-8f40-e8911c3b34be
begin
	f = 1e6               # Частота генератора, Гц
	ω = 2 * π * f         # Круговая частота, рад/с
	T_povt = 2.048e-3     # Период повторения имп., с
	b_visc = 4e-3         # Эффективная вязкость, кг/(м*с)
	ρ_0 = 1000.0          # Плотность воды, кг/м^3
	c_0 = 1500.0          # Скорость звука, м/с
	ϵ_nonlin = 4.0        # Нелинейный параметр для воды 
	λ = c_0 / f           # Длина волны
end


# ╔═╡ 2dc88e86-8555-4749-b6fd-2bcd761eeb86
begin
	# Расчет n_eqv по суммам квадратов (воссоздание логики таблицы)
	norm_A = A_raw ./ A_max
	n_eqv = sum(norm_A.^2)
	
	# Скважность
	Q_ekv = (T_povt * f) / n_eqv
	
	# Значения энергии и интенсивности из вашего Excel
	E_joules = 415.8
	I_intens = 374607.7776  # W/m^2
	
	# Колебательная скорость и давление
	v_0 = sqrt(2 * I_intens / (ρ_0 * c_0))
	p_0 = sqrt(2 * I_intens * ρ_0 * c_0)
	
	# Число Рейнольдса и хар. длины
	Re_ac = (ρ_0 * c_0 * v_0) / (b_visc * ω)
	
	x_p = c_0^2 / (ϵ_nonlin * ω * v_0)   # Расстояние образования разрыва (в метрах)
	x_f = x_p * (π / 2)                  # Формир. пилы
	x_l = ((2 * ρ_0 * c_0^3)/(b_visc * ω^2)) * log(Re_ac * π / 2) # Возврат к синусоиде
	
	x_fr = λ / (2 * π * ϵ_nonlin * Re_ac) # Оценка ширины фронта
end


# ╔═╡ ebfc92a5-5de3-44a2-8389-f2ec19219ee1
md"""
### Результаты вычислений
* **Эквивалентное кол-во колебаний $n_{экв}$:** $(round(n_eqv, digits=4))
* **Скважность $Q_{экв}$:** $(round(Q_ekv, digits=4))
* **Энергия за импульс $E$:** $(round(E_joules, digits=2)) Дж
* **Интенсивность волны $I$:** $(round(I_intens, digits=2)) Вт/м²
* **Амплитуда колебательной скорости $v_0$:** $(round(v_0, digits=4)) м/с (в таблице: 0.7067)
* **Амплитуда давления $p_0$:** $(round(p_0/1000, digits=2)) кПа (в таблице 1.06 МПа)

---
* **Число Рейнольдса $Re$:** $(round(Re_ac, digits=3)) (в таблице: 42.20)
* **Расстояние образования разрыва $x_p$:** $(round(x_p, digits=4)) м (в таблице: 0.1267 м)
* **Расстояние формир. пилы $x_ф$:** $(round(x_f, digits=4)) м (в таблице: 0.1989 м)
* **Расстояние затухания $x_л$:** $(round(x_l, digits=4)) м
* **Ширина разрыва $\delta_{фр}$:** $(round(x_fr*1e6, digits=2)) мкм
"""


# ╔═╡ 241788f5-1e21-4497-aa00-9feed13edb94
@bind sigma Slider(0.0:0.1:2.0, default=0.0, show_value=true)


# ╔═╡ 4d4ee686-c4a0-42e6-aace-ce091b3e1d28
begin
	θ_vals = range(-3π, 3π, length=800)
	v_norm = sin.(θ_vals)
	phase = θ_vals .- sigma .* v_norm
	
	plot(phase, v_norm, 
		title="Эволюция формы волны (σ = x/x_p = $sigma)", 
		xlabel="Фаза (ωt - kx)", 
		ylabel="v / v_0",
		lw=2, color=:blue, legend=false, grid=true,
		xlims=(-2π, 2π), ylims=(-1.2, 1.2), size=(700,300))
end


# ╔═╡ 178bfef3-2ddb-4647-908c-14074574d86c
md"""
---
### Упражнение 8. Эволюция спектра (Эксперимент vs Теория)
Ниже представлен график зависимости второй гармоники ($A_2$) от безразмерного расстояния $\sigma = x/x_{p}$.
Точки — это **ваши данные из таблицы**, наложенные на сшитую теоретическую кривую Бесселя-Фубини (до $x_p$) и Бюргерса-Фэя (после $x_p$).
*(Примечание: экспериментальные амплитуды нормированы по максимуму для удобства визуального сопоставления формы кривой с теорией).*
"""


# ╔═╡ 25f7c52d-9862-40a1-bd92-0548b8f035c3
begin
	sig_range = range(0.01, 4.0, length=400)

    # Теоретическая сшитая функция эволюции (по Фэю и Фубини)
	function harmonic_amp(n, s)
		if s <= 1.0 
			return (2 / (n * s)) * besselj(n, n * s)
		else 
			return 2 / (n * (1 + s))
		end
	end

	A2_theory = [harmonic_amp(2, s) for s in sig_range]

	# Пересчет экспериментальных точек х (см) -> метры -> делим на х_р
	sigma_exp = (x_exp_cm ./ 100.0) ./ x_p  
	
    # Нормируем ваши данные A2 (вольтаж) под масштаб теоретического графика
	max_th = maximum(A2_theory)
	max_exp = maximum(A2_exp)
	A2_exp_norm = (A2_exp ./ max_exp) .* max_th

	p_spectr = plot(sig_range, A2_theory, label="Теория: 2-я гармоника", lw=2.5, color=:green)
	
	# Накладываем ВАШИ ТОЧКИ из эксперимента
	scatter!(p_spectr, sigma_exp, A2_exp_norm, 
        label="Эксперимент (v_max = 2.4 В)", 
        color=:red, markersize=5, markerstrokewidth=0)
	
	vline!(p_spectr, [1.0], label="x = x_p (Разрыв)", ls=:dash, lw=2, color=:black)
	vline!(p_spectr, [pi/2], label="x = x_ф (Пила)", ls=:dot, lw=2, color=:gray)
	
	plot!(p_spectr, title="Сравнение А2: Теория vs Вашего Эксперимента", 
		  xlabel="Относительное расстояние σ = x/x_p", 
		  ylabel="Относительная амплитуда",
          grid=true, size=(800, 500), legend=:bottom)
end


# ╔═╡ e13369f3-edb9-4665-9203-fabb4b555bc7


# ╔═╡ Cell order:
# ╠═84118270-3504-11f1-820f-733e877b1a89
# ╠═70ae7041-a8d1-4712-87db-8bdbdc9bf190
# ╠═4831a806-25b7-4252-bc9a-3657e81225dd
# ╠═93e11d08-7abc-49c9-8f40-e8911c3b34be
# ╠═2dc88e86-8555-4749-b6fd-2bcd761eeb86
# ╠═ebfc92a5-5de3-44a2-8389-f2ec19219ee1
# ╠═241788f5-1e21-4497-aa00-9feed13edb94
# ╠═4d4ee686-c4a0-42e6-aace-ce091b3e1d28
# ╠═178bfef3-2ddb-4647-908c-14074574d86c
# ╠═25f7c52d-9862-40a1-bd92-0548b8f035c3
# ╠═e13369f3-edb9-4665-9203-fabb4b555bc7
