### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f9710f40-1f10-11f1-b0a2-1fffd7e1db7c
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add("Plots")
    using Plots
    default(grid=true, minorgrid=true, linewidth=2, markershape=:circle, fontfamily="sans-serif", legend=:topright)
end


# ╔═╡ 4488e990-415d-49a2-82a4-ab09242d5915
begin
    function normalize_dB(A, pos)
        return A .- (pos .* 6.02)
    end
end


# ╔═╡ 5a55b956-f152-47c2-927d-5767f48de81e
md"### Упражнение 1. Изучение теплового шума и шумовых параметров усилителя"


# ╔═╡ 05f417ef-f9fe-4366-9d75-14f440e63dc3
begin

    R_kOm_raw = [5100.0, 2360.0, 964.0, 470.0, 185.0, 99.0, 42.0, 22.0, 9.1, 0.0]
    A_dB_T_raw = [-43.87, -50.08, -57.22, -61.22, -67.53, -67.41, -64.91, -64.09, -65.95, -65.49]
    pos_T_raw = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

    R_kOm = reverse(R_kOm_raw)
    A_dB_T = reverse(A_dB_T_raw)
    pos_T = reverse(pos_T_raw)

    A_norm_T = normalize_dB(A_dB_T, pos_T)
end


# ╔═╡ 54531a61-0bd4-4437-a4a1-e4f31c0cc525
plot(R_kOm[2:end], A_norm_T[2:end], 
    xscale=:log10, 
    xlabel="R kOm", ylabel="A dB", 
    title="Зависимость мощности шума от величины\nсопротивления", 
    label="", color=:blue, markershape=:circle)


# ╔═╡ 8a5bfafd-fc3f-4f57-a29d-f6c19dc156a1
begin
    S_rel_T = 10 .^ (A_norm_T ./ 10.0)
    
    delta_S = S_rel_T[6] - S_rel_T[1]
    kT4 = delta_S / R_kOm[6]
    
    K_noise = S_rel_T[2:end] ./ (kT4 .* R_kOm[2:end])
    
    plot(R_kOm[2:8], K_noise[1:7], 
        xlabel="R kOm", ylabel="K", 
        title="Зависимость коэффициента шума от\nсопротивления", 
        label="", color=:red, markershape=:circle)
end


# ╔═╡ 133296ea-5305-44bc-8997-79166d67deb2
begin
    Is_mkA = [1.3, 2.5, 5.2, 7.4, 13.0, 25.0, 47.0, 100.0, 175.0, 225.0]
    
    A_dB_S = [-48.27, -46.67, -43.13, -43.25, -41.43, -39.96, -40.99, -42.03, -40.64, -40.34]
    pos_S  = [3, 3, 3, 3, 3, 3, 2, 1, 1, 1]
    
    A_norm_S = normalize_dB(A_dB_S, pos_S)
end


# ╔═╡ 92785006-2ec4-4328-bc0e-82a60c15d6df
plot(Is_mkA, A_norm_S, 
    xscale=:log10, 
    xlabel="Is mkA", ylabel="A dB", 
    title="Дробовой шум, зависимость от I", 
    label="", color=:purple)


# ╔═╡ c6ad7b0a-f232-4339-a1d6-a1770b5cc575
begin
    Ie_mkA = [0.0, 170.0, 200.0, 245.0, 295.0, 335.0, 385.0, 470.0, 550.0, 630.0, 710.0]
    
    A_dB_E = [-50.62, -53.86, -54.52, -55.41, -55.41, -53.70, -49.12, -43.38, -51.79, -48.35, -46.54]
    pos_E  = [4,     3,     3,     3,     3,    3,     3,     3,     1,     1,     1]
    
    A_norm_E = normalize_dB(A_dB_E, pos_E)
end


# ╔═╡ f0177fbe-32af-408c-b468-19fb06f4c147

plot(Ie_mkA[2:end], A_norm_E[2:end], 
    xscale=:log10, 
    xlabel="Ie mkA", ylabel="A dB", 
    title="Зависимость избыточного шума от Ie", 
    label="", color=:green)


# ╔═╡ Cell order:
# ╟─f9710f40-1f10-11f1-b0a2-1fffd7e1db7c
# ╟─4488e990-415d-49a2-82a4-ab09242d5915
# ╟─5a55b956-f152-47c2-927d-5767f48de81e
# ╠═05f417ef-f9fe-4366-9d75-14f440e63dc3
# ╟─54531a61-0bd4-4437-a4a1-e4f31c0cc525
# ╠═8a5bfafd-fc3f-4f57-a29d-f6c19dc156a1
# ╟─133296ea-5305-44bc-8997-79166d67deb2
# ╟─92785006-2ec4-4328-bc0e-82a60c15d6df
# ╠═c6ad7b0a-f232-4339-a1d6-a1770b5cc575
# ╟─f0177fbe-32af-408c-b468-19fb06f4c147
