### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 693d3094-2c52-11f1-b2c9-f375929df256
begin
    using Pkg
    Pkg.add(["DataFrames", "Plots"])
    using DataFrames
    using Plots
end


# ╔═╡ ef8a4d14-6161-484f-81a8-428699977ff5
begin
    L1_uH = 412.0          # Индуктивность контура (по методичке 412 мкГн)
    f_res_kHz = 670.7      # Резонансная частота f1 (f_н / 2) в кГц
    
    C_0V_pF = 100.0        # Показания шкалы конденсатора при смещении 0 В (пФ)
    C_minus01V_pF = 130.0  # Показания при смещении -0.1 В (пФ)
    # ------------------------------

    # Расчеты (перевод в системы СИ внутри)
    L1 = L1_uH * 1e-6
    f_res = f_res_kHz * 1e3
    
    # 1. Расчет полной емкости контура C_k
    ω1 = 2 * π * f_res
    C_k_F = 1 / (ω1^2 * L1)
    C_k_pF = C_k_F * 1e12
    
    # 2. Изменение емкости 
    delta_C_pF = C_0V_pF - C_minus01V_pF
    delta_C_F = delta_C_pF * 1e-12
    
    # 3. Коэффициент модуляции емкости при U_н = 0.1 В
    m1 = delta_C_F / (2 * C_k_F)
    
    md"""
	### Результаты Упражнения 1:
	* Полная емкость контура $C_k$: **$(round(C_k_pF, digits=2)) пФ**
	* Изменение емкости $\Delta C$: **$(round(delta_C_pF, digits=2)) пФ**
	* Коэффициент модуляции $m_1$ (при 0.1 В): **$(round(m1, digits=4))**
	"""
end


# ╔═╡ ee7c9d5b-4d2b-44c7-9518-7106148accc1
begin
    U_voltmeter_V = 0.1   # Напряжение по вольтметру (показания с клемм х5)
    delta_f_kHz = 30      # Полоса пропускания контура (на уровне 0.7) в кГц
    # ------------------------------

    # 1. Расчет истинной амплитуды накачки U_н
    # Делим на 5 (усилитель) и умножаем на sqrt(2) (переход от эфф. к амплитуде)
    U_n_amp = (U_voltmeter_V / 5.0) * sqrt(2)
    
    # 2. Расчет m при критическом напряжении (линейная аппроксимация)
    m_crit = m1 * (U_n_amp / 0.1)
    
    # 3. Расчет добротности Q
    Q_factor = f_res_kHz / delta_f_kHz
    
    # 4. Проверка условия Q * m
    Qm_product = Q_factor * m_crit
    
    md"""
    ### Результаты Упражнения 2:
    * Истинная амплитуда накачки $U_н$: **$(round(U_n_amp, digits=4)) В**
    * Добротность контура $Q$: **$(round(Q_factor, digits=1))**
    * Критический коэффициент модуляции $m$: **$(round(m_crit, digits=4))**
    * Произведение $Q \cdot m$: **$(round(Qm_product, digits=3))** (В теории должно быть $\approx 1$)
    """
end


# ╔═╡ d6ba58e5-5aa5-4af0-835b-1967fd1f81ca
begin
    data_ex3 = DataFrame(
        Frequency5_kHz = [334.0, 333.0, 334.5, 333.8, 335.5, 336, 336.5, 337],
        Amplitude_K5  = [400,  200,  600,  300,  400,  300, 200, 200],  # Для усиления K = 5
		Frequency10_kHz = [334.0, 334.5, 334.3, 334.7, 335.1, 335.2, 335.4, 333.6],
        Amplitude_K10 = [200, 400, 300, 700, 600, 400, 300, 200]   # Для усиления K = 10
    )
end


# ╔═╡ 092a6c9b-38cd-4ae8-984a-ae216cc14bd4
begin
    data_ex4 = DataFrame(
        Frequency5_kHz = [199.8, 199.9, 199.7, 199.6, 199.5, 200.2, 200.3],
		Frequency10_kHz = [199.9, 199.8, 199.7, 199.6, 200.1, 200.2, 200.3],
        Amplitude_K5  = [150, 200, 100, 70, 60, 80, 70],  
        Amplitude_K10 = [150, 80, 60, 70, 80, 50, 40]   
    )
end


# ╔═╡ 62ce9f5f-3020-422e-95f2-c16dd848f715
begin
    data_ex5 = DataFrame(
        Frequency5_kHz = [470.0, 470.1, 470.2, 469.9, 469.8],
        Amplitude_K5  = [300, 100, 40, 80, 40],  
    )
	data_ex5_2 = DataFrame(
		Frequency10_kHz = [470.0, 470.1, 470.2, 470.3, 470.4, 469.9, 469.8, 469.7, 469.6],
		Amplitude_K10 = [200, 150, 80, 60, 40, 125, 80, 60, 40]
	)
end


# ╔═╡ d9b6d338-4319-4f33-ba35-9598b1ed2b0b
begin
	scatter(data_ex3.Frequency5_kHz, data_ex3.Amplitude_K5, 
	    label="K = 5", marker=:circle, linewidth=2, 
	    xlabel="Частота (кГц)", ylabel="Амплитуда", 
	    title="Оконтурный параметрический усилитель", 
	    grid=true, legend=:topright)
	scatter!(data_ex3.Frequency10_kHz, data_ex3.Amplitude_K10, 
    	label="K = 10", marker=:square, linewidth=2)
end


# ╔═╡ fc173783-29a1-4639-8768-0a5d563f8250
begin
	scatter(data_ex4.Frequency5_kHz, data_ex4.Amplitude_K5, 
	    label="K = 5", marker=:circle, linewidth=2, color=:blue,
	    xlabel="Частота (кГц)", ylabel="Амплитуда", 
	    title="Двухконтурный регенеративный усилитель", 
	    grid=true)
	scatter!(data_ex4.Frequency10_kHz, data_ex4.Amplitude_K10, 
	    label="K = 10", marker=:square, linewidth=2, color=:red)
end

# ╔═╡ fd044189-f45a-47bf-adb6-e397213afda8
begin
	scatter(data_ex5.Frequency5_kHz, data_ex5.Amplitude_K5, 
	    label="K = 5", marker=:circle, linewidth=2, color=:green,
	    xlabel="Частота комбинационная (кГц)", ylabel="Амплитуда", 
	    title="Усилитель-преобразователь", 
	    grid=true)
	scatter!(data_ex5_2.Frequency10_kHz, data_ex5_2.Amplitude_K10, 
	    label="K = 10", marker=:square, linewidth=2, color=:orange)
end

# ╔═╡ Cell order:
# ╠═693d3094-2c52-11f1-b2c9-f375929df256
# ╠═ef8a4d14-6161-484f-81a8-428699977ff5
# ╠═ee7c9d5b-4d2b-44c7-9518-7106148accc1
# ╠═d6ba58e5-5aa5-4af0-835b-1967fd1f81ca
# ╠═092a6c9b-38cd-4ae8-984a-ae216cc14bd4
# ╠═62ce9f5f-3020-422e-95f2-c16dd848f715
# ╟─d9b6d338-4319-4f33-ba35-9598b1ed2b0b
# ╟─fc173783-29a1-4639-8768-0a5d563f8250
# ╠═fd044189-f45a-47bf-adb6-e397213afda8
