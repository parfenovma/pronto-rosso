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
"""


# ╔═╡ 4831a806-25b7-4252-bc9a-3657e81225dd
begin
	A_max = 2.3
	A_raw = [0.5, 1, 1.3, 1.5, 1.9, 2.1, 2.2, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.2, 2.1, 1.9, 1.7, 1.6, 1.4, 1.1, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
	
	x_exp_cm = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]
	A2_exp = [50, 55, 60, 100, 120, 140, 200, 240, 280, 300, 330, 350, 350, 340, 330, 320, 310, 300, 290, 280, 270, 250, 230, 220]
end


# ╔═╡ 93e11d08-7abc-49c9-8f40-e8911c3b34be
begin
	f = 1e6
	ω = 2 * π * f
	T_povt = 2.048e-3
	b_visc = 4e-3
	ρ_0 = 1000.0
	c_0 = 1500.0
	ϵ_nonlin = 4.0
	λ = c_0 / f
end

# ╔═╡ 2dc88e86-8555-4749-b6fd-2bcd761eeb86
begin
	norm_A = A_raw ./ A_max
	n_eqv = sum(norm_A.^2)
	
	Q_ekv = (T_povt * f) / n_eqv

	delta_T = 0.8
	t_izl = 180.0 
	V_k = 110e-6
	d_k = 0.03
	S_k = π * (d_k / 2)^2
	
	E_joules = 4200 * ρ_0 * V_k * delta_T
	I_intens = (E_joules * Q_ekv) / (S_k * t_izl)
	
	v_0 = sqrt(2 * I_intens / (ρ_0 * c_0))
	p_0 = sqrt(2 * I_intens * ρ_0 * c_0)
	
	Re_ac = (ρ_0 * c_0 * v_0) / (b_visc * ω)
	
	x_p = c_0^2 / (ϵ_nonlin * ω * v_0)
	x_f = x_p * (π / 2)
	x_l = ((2 * ρ_0 * c_0^3)/(b_visc * ω^2)) * log(Re_ac * π / 2)
	
	x_fr = λ / (2 * π * ϵ_nonlin * Re_ac)
end


# ╔═╡ ebfc92a5-5de3-44a2-8389-f2ec19219ee1
md"""
### Результаты вычислений
* **Эквивалентное кол-во колебаний $n_{экв}$:** $(round(n_eqv, digits=4))
* **Скважность $Q_{экв}$:** $(round(Q_ekv, digits=4))
* **Энергия $E$:** $(round(E_joules, digits=2)) Дж
* **Интенсивность $I$:** $(round(I_intens, digits=2)) Вт/м²
* **Амплитуда (по скорости) $v_0$:** $(round(v_0, digits=4)) м/с
* **Амплитуда (по давлению) $p_0$:** $(round(p_0/1000, digits=2)) кПа

---
* **Число Рейнольдса $Re$:** $(round(Re_ac, digits=3))
* **Разрыв $x_p$:** $(round(x_p, digits=4)) м
* **Пила $x_ф$:** $(round(x_f, digits=4)) м
* **Затухание $x_л$:** $(round(x_l, digits=4)) м
* **ширина разрыва $\delta_{фр}$:** $(round(x_fr*1e6, digits=2)) мкм
"""


# ╔═╡ 241788f5-1e21-4497-aa00-9feed13edb94
# @bind sigma Slider(0.0:0.1:2.0, default=0.0, show_value=true)


# ╔═╡ 4d4ee686-c4a0-42e6-aace-ce091b3e1d28
# begin
# 	θ_vals = range(-3π, 3π, length=800)
# 	v_norm = sin.(θ_vals)
# 	phase = θ_vals .- sigma .* v_norm
	
# 	plot(phase, v_norm, 
# 		title="Эволюция формы волны (σ = x/x_p = $sigma)", 
# 		xlabel="Фаза (ωt - kx)", 
# 		ylabel="v / v_0",
# 		lw=2, color=:blue, legend=false, grid=true,
# 		xlims=(-2π, 2π), ylims=(-1.2, 1.2), size=(700,300))
# end


# ╔═╡ 178bfef3-2ddb-4647-908c-14074574d86c
md"""
---
### Упражнение 8. Эволюция спектра
"""


# ╔═╡ 25f7c52d-9862-40a1-bd92-0548b8f035c3
begin
	sig_range = range(0.01, 4.0, length=200)

	function harmonic_amp(n, s)
		if s <= 1.0 
			return (2 / (n * s)) * besselj(n, n * s)
		else 
			return 2 / (n * (1 + s))
		end
	end

	A2_theory = [harmonic_amp(2, s) for s in sig_range]


	sigma_exp = (x_exp_cm ./ 100.0) ./ x_p  
	
	max_th = maximum(A2_theory)
	max_exp = maximum(A2_exp)
	A2_exp_norm = (A2_exp ./ max_exp) .* max_th

	p_spectr = scatter(sigma_exp, A2_exp_norm, 
        label="Эксперимент (v_max = 2.3 В)", 
        color=:red, markersize=5, markerstrokewidth=0)
	
	# scatter!(p_spectr, sigma_exp, A2_exp_norm, 
 #        label="Эксперимент (v_max = 2.3 В)", 
 #        color=:red, markersize=5, markerstrokewidth=0)
	
	vline!(p_spectr, [1.0], label="x = x_p (Разрыв)", ls=:dash, lw=2, color=:black)
	vline!(p_spectr, [pi/2], label="x = x_ф (Пила)", ls=:dot, lw=2, color=:gray)
	
	plot!(p_spectr, title="Сравнение А2", 
		  xlabel="Относительное расстояние σ = x/x_p(теор)", 
		  ylabel="Относительная амплитуда",
          grid=true, size=(800, 500), legend=:bottom)
end


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
