### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ c7528e18-2c52-11f1-9801-3fd6454e1d64
begin
	import Pkg
	Pkg.add(["DataFrames", "Plots"])
	using DataFrames, Plots
	
	# Настройки графиков по умолчанию
	default(grid=true, linewidth=2, marker=:circle, markersize=4, fontfamily="helvetica")
	md"**Библиотеки загружены!**"
end


# ╔═╡ 7e8a59f1-2e22-492e-8140-f15c0aed0b98
begin
	R1 = 3000.0   # Ом
	R2 = 3000.0   # Ом
	C1 = 22e-9    # Фарад
	C2 = 22e-9    # Фарад
	
	f_0_theor = 1 / (2 * pi * sqrt(R1 * R2 * C1 * C2))
	
	md"""
	**Теоретические расчеты:**
	* Резонансная частота $f_0$: **$(round(f_0_theor, digits=1)) Гц**
	"""
end


# ╔═╡ 2b697f55-6757-490b-9789-20e86f7599a1
begin
	df_ex1 = DataFrame(
		f_Hz  = [200, 1000, 1500, 2000, 4500, 6000, 8000, 10000], # Частота генератора
		U_in  = [1.0, 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0],   # Напряжение на входе (В)
		U_out = [0.184,0.258, 0.306, 0.324, 0.310, 0.278, 0.248, 0.216]   # Напряжение на выходе (В)
	)
	
	# Расчет коэффициента передачи
	df_ex1.K = df_ex1.U_out ./ df_ex1.U_in
	
	# # Поиск максимума
	# max_K_index = argmax(df_ex1.K)
	# f_01_exp = df_ex1.f_Hz[max_K_index]
	# K_max = df_ex1.K[max_K_index]
	
	plot1 = plot(df_ex1.f_Hz, df_ex1.K, 
		xaxis=:log10, 
		xlabel="Частота f, Гц (лог. масштаб)", 
		ylabel="Коэффициент передачи K",
		title="АЧХ цепочки Вина",
		# label="Эксперимент",
				 color=:blue)
	
	# scatter!(plot1, [f_01_exp], [K_max], color=:red, label="Максимум ($f_01_exp Гц)")
	
	plot1
end


# ╔═╡ 4f5e568f-3384-42c6-900e-6fbe9bcb0c82
begin
	df_ex3 = DataFrame(
		f1_Hz = [400, 1000, 1200, 1500, 2000, 2200, 2500, 2800, 2900, 3100, 3500, 4000, 6000, 8000, 10000],
		f2_Hz = [400, 1000, 1500, 2000, 2200, 2500, 2700, 2900, 3000, 3400, 3800, 4500, 6000, 8000, 10000],
		f3_Hz = [400, 1000, 1500, 2000, 2300, 2600, 2700, 3000, 3500, 4000, 5400, 5401, 6000, 8000, 10000],
		U_alpha1 = [0.32, 0.64, 0.72, 0.96, 1.2, 1.52, 2.08, 2.4, 2.4, 2.4, 2, 1.6, 0.8, 0.72, 0.64],
		U_alpha2 = [0.4, 0.8, 1.12, 1.68, 1.84, 3.84, 5.68, 6.80, 6.72, 3.68, 2.4, 1.52, 0.9, 0.8, 0.7],
		U_alpha3 = [0.4, 0.6, 0.8, 1.12, 1.36, 1.52, 1.6, 1.68, 1.52, 1.28, 0.96, 0.96, 0.8, 0.6, 0.5],
	)
	
	# Нормировка (U / U_max)
	norm_U1 = df_ex3.U_alpha1 ./ maximum(df_ex3.U_alpha1)
	norm_U2 = df_ex3.U_alpha2 ./ maximum(df_ex3.U_alpha2)
	norm_U3 = df_ex3.U_alpha3 ./ maximum(df_ex3.U_alpha3)
	
	plot3 = plot(df_ex3.f1_Hz, norm_U1, label="α=0.7", xlabel="Частота f, Гц", ylabel="U / U_max", title="Нормированные резонансные кривые")
	plot!(plot3, df_ex3.f2_Hz, norm_U2, label="α=0.9")
	plot!(plot3, df_ex3.f3_Hz, norm_U3, label="α=0.6")
	# plot!(plot3, df_ex3.f_Hz, norm_U3, label="α=0.6")
	
	plot3
end


# ╔═╡ b1c13032-5d90-4709-b21d-0fdef5ee84b6
begin
	# === ВВОД ДАННЫХ ===
	df_ex4 = DataFrame(
		alpha_norm = [0.98, 0.75, 0.65, 0.5], # Отношение α/α_кр
		N = [1, 2, 3, 5],                    # Число периодов, через которое проводилось измерение
		X_n = [4.0, 4.0, 4.0, 4.0],         # Начальная амплитуда по сетке осциллографа
		X_nN = [1.2, 1.5, 2.0, 2.8]         # Амплитуда через N периодов
	)
	
	# Расчет декремента и добротности (Q = pi/d)
	df_ex4.d = [0.09, 1.39, 2.25, 2.48] # (1 ./ df_ex4.N) .* log.(df_ex4.X_n ./ df_ex4.X_nN)
	df_ex4.Q = pi ./ df_ex4.d
	
	p_d = plot(df_ex4.alpha_norm, df_ex4.d, color=:blue, label="Декремент затухания d", ylabel="d")
	# p_q = plot(df_ex4.alpha_norm, df_ex4.Q, color=:red, label="Добротность Q", ylabel="Q")
	
	# plot_4 = plot(p_d, layout=(2, 1), xlabel="α / α_кр", title="Зависимость d и Q от регенерации")
	p_d
end


# ╔═╡ f7fbd3a7-646a-46e3-a716-53eebebba6c2
begin
	# === ВВОД ДАННЫХ ===
	df_ex7 = DataFrame(
		V_in = [1.4, 2.2, 2.8, 3.0, 3.6, 4.6, 5.4],  # Амплитуда на входе
		U_out = [1.6, 2.4, 3.0, 3.2, 4.0, 5.0, 5.6]  # Амплитуда на выходе
	)
	
	# Расчет коэффициента усиления от амплитуды
	df_ex7.K_amp = df_ex7.U_out ./ df_ex7.V_in
	
	p7_1 = plot(df_ex7.V_in, df_ex7.U_out, 
		xlabel="Входной сигнал V_0, В", ylabel="Выходной сигнал U_0, В", 
		title="Амплитудная характеристика U_0(V_0)", label="U_out")
	
	p7_2 = plot(df_ex7.V_in, df_ex7.K_amp, 
		xlabel="Входной сигнал V_0, В", ylabel="Коэф. усиления K", 
		title="Зависимость K(V_0)", label="K(U)", color=:orange)
	
	plot7 = plot(p7_1, p7_2, layout=(1,2), size=(800, 400))
	plot7
end


# ╔═╡ Cell order:
# ╠═c7528e18-2c52-11f1-9801-3fd6454e1d64
# ╠═7e8a59f1-2e22-492e-8140-f15c0aed0b98
# ╠═2b697f55-6757-490b-9789-20e86f7599a1
# ╠═4f5e568f-3384-42c6-900e-6fbe9bcb0c82
# ╠═b1c13032-5d90-4709-b21d-0fdef5ee84b6
# ╠═f7fbd3a7-646a-46e3-a716-53eebebba6c2
