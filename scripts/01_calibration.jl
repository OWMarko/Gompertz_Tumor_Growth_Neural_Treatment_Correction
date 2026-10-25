ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots, Printf

default(; margin=5Plots.mm, gridalpha=0.2, legendfontsize=11,
          tickfontsize=10, guidefontsize=12, titlefontsize=13)

include(joinpath(@__DIR__, "..", "src", "physics.jl"))
include(joinpath(@__DIR__, "..", "src", "tools.jl"))

function main()

    dossier = joinpath(@__DIR__, "..", "resultats")
    mkpath(dossier)

    # On estime la croissance a partir des observations sans traitement
    donnees = creer_donnees(0.0:0.5:20.0, Float64[], 41)
    ajustement = calibrer_croissance(donnees)
    r, K = ajustement.r, ajustement.K

    N = reference_analytique(donnees.t, Float64[], r, K, 0.0)
    numerique = simuler_parametrique(0.0, donnees.t, Float64[], r, K)

    erreur = maximum(abs, numerique.N .- N)

    # On conserve les parametres pour l'experience avec traitement
    parametres = Dict("r"=>r, "K"=>K, "N0"=>0.1, "k"=>0.25,
                      "graine_donnees"=>41, "bruit_relatif"=>0.03,
                      "julia"=>string(VERSION), "erreur_integration"=>erreur,
                      "convergence"=>Optim.converged(ajustement.resultat))

    open(joinpath(dossier, "parametres_croissance.toml"), "w") do io
        TOML.print(io, parametres)

    end

    lignes = [[donnees.t[i], donnees.reference[i], donnees.y[i], N[i]] for i in eachindex(N)]

    ecrire_csv(joinpath(dossier, "calibration.csv"), ["t_jour", "reference", "observation", "calibration"], lignes)

    p = plot(donnees.t, donnees.reference; label="Reference synthetique",
             color=:black, linestyle=:dash, xlabel="Temps (jours)",
             ylabel="Volume normalise", title="1. Calibration sans traitement")

    scatter!(p, donnees.t, donnees.y; label="Observations, bruit 3 %", ms=3)
    plot!(p, donnees.t, N; label="Gompertz calibre", lw=2, color=:darkgreen)
    savefig(p, joinpath(dossier, "01_calibration.png"))

    @printf("r = %.8f /jour ; K = %.8f ; erreur integration = %.3e\n", r, K, erreur)
    println("Convergence L-BFGS : ", Optim.converged(ajustement.resultat))

end

main()
