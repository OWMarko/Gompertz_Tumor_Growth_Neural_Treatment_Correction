ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots, Printf

default(; margin=5Plots.mm, gridalpha=0.2, legendfontsize=11,
          tickfontsize=10, guidefontsize=12, titlefontsize=13)

include(joinpath(@__DIR__, "..", "src", "physics.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "tools.jl"))

function main()

    dossier = joinpath(@__DIR__, "..", "resultats")
    fichier = joinpath(dossier, "parametres_croissance.toml")
    isfile(fichier) || error("Lancer d'abord scripts/01_calibration.jl.")

    croissance = TOML.parsefile(fichier)

    r, K = croissance["r"], croissance["K"]
    N0, k = croissance["N0"], croissance["k"]

    alpha, iterations = 1e-6, 200
    graines = [11, 22, 33]
    temps = 0.0:0.5:60.0

    noms = ["apprentissage", "validation", "test_A", "test_B"] # 1 calendrier pour apprendre, 1 pour choisir les poids, 2 pour tester

    dates = [[10.0,20.0,30.0], [12.0,28.0,44.0],
             [15.0,35.0,50.0], [8.0,24.0,46.0]]

    donnees = [creer_donnees(temps, dates[i], 41+i; N0, k) for i in 1:4]

    apprentissage, validation = donnees[1], donnees[2]
    traitement = calibrer_traitement(apprentissage, r, K)
    delta = traitement.delta

    @printf("Traitement parametrique : delta = %.8f /jour\n", delta)

    # entrainement des trois initialisations sur les memes obs
    reseaux = []
    historique = []
    for graine in graines

        res = entrainer_ude(initialiser_reseau(graine), apprentissage, validation, r, K; alpha, iterations)
        push!(reseaux, res)

        for ligne in res.historique

            push!(historique, [graine; ligne])

        end
    end

    ecrire_csv(joinpath(dossier, "pertes.csv"), ["graine", "iteration", "mse_apprentissage", "mse_validation", "perte_penalisee"], historique)

    poids = reduce(hcat, [res.p for res in reseaux])

    ecrire_csv(joinpath(dossier, "poids_ude.csv"), ["parametre"; ["graine_$s" for s in graines]], [[i; poids[i,:]] for i in axes(poids, 1)])

    # comparaison des 2 modeles parametriques et les 3 reseaux

    metriques, predictions, controles = [], [], []
    courbes = []

    modeles = vcat(["gompertz", "parametrique"], fill("ude", length(graines)))
    graines_modeles = vcat([0, 0], graines)

    for (nom, d) in zip(noms, donnees)

        gomp = reference_analytique(d.t, Float64[], r, K, 0.0; N0, k)
        param = reference_analytique(d.t, d.dates, r, K, delta; N0, k)
        solutions = [(N=gomp, statut="Analytique", pas=0),(N=param, statut="Analytique", pas=0)]

        for (graine, res) in zip(graines, reseaux)

            sol = simuler_ude(res.p, d.t, d.dates, r, K; N0, k)
            fin = simuler_ude(res.p, d.t, d.dates, r, K; N0, k, abstol=1e-12, reltol=1e-10)
            ecart = maximum(abs, sol.N .- fin.N)
            ecart < 1e-6 || error("Predictions sensibles aux tolerances : $nom.")

            push!(controles, [graine, nom, ecart])
            push!(solutions, sol)

        end

        for (modele, graine, sol) in zip(modeles, graines_modeles, solutions)

            e = scores(sol.N, d)
            push!(metriques, [modele, graine, nom, e.mse_observations, e.rmse_reference, e.rmse_log, sol.statut, sol.pas])

        end

        for i in eachindex(d.t)

            C = concentration(d.t[i], d.dates, k)
            ligne = Any[nom, d.t[i], C, d.reference[i], d.y[i]]

            for sol in solutions

                push!(ligne, sol.N[i])

            end

            push!(predictions, ligne)

        end

        push!(courbes, (gomp=gomp, param=param, ude=solutions[3:end]))
        
    end


    ecrire_csv(joinpath(dossier, "metriques.csv"), ["modele", "graine", "protocole", "mse_observations", "rmse_reference", "rmse_log_reference", "statut", "pas_acceptes"], metriques)


    ecrire_csv(joinpath(dossier, "predictions.csv"), ["protocole", "t_jour", "C", "reference", "observation", "gompertz", "parametrique", ["ude_$s" for s in graines]...], predictions)


    ecrire_csv(joinpath(dossier, "controle_numerique.csv"), ["graine", "protocole", "ecart_max_tolerances"], controles)


    config = Dict("r_est"=>r, "K_est"=>K, "delta_est"=>delta, "alpha"=>alpha,
                  "N0"=>N0, "k"=>k, "bruit_relatif"=>0.03,
                  "convergence_delta"=>Optim.converged(traitement.resultat),
                  "iterations_delta"=>Optim.iterations(traitement.resultat),
                  "iterations_max"=>iterations, "graines_reseau"=>graines,
                  "graines_donnees"=>[42,43,44,45], "abstol"=>1e-10, "reltol"=>1e-8,
                  "julia"=>string(VERSION), "architecture"=>[2,8,1],
                  "iterations_retenues"=>[res.iteration for res in reseaux],
                  "iterations_effectuees"=>[res.iterations for res in reseaux],
                  "convergences"=>[res.converge for res in reseaux])

    for (nom, calendrier) in zip(noms, dates)

        config["dates_$nom"] = calendrier

    end

    open(joinpath(dossier, "configuration.toml"), "w") do io 
        TOML.print(io, config)
    end

    # Figures des ajustement, pertes, tests et correction sur la reference
    p1 = plot(apprentissage.t, apprentissage.reference; color=:black,
              linestyle=:dash, label="Reference", ylabel="Volume normalise",
              xlabel="Temps (jours)",
              title="2. Ajustement sous traitement")

    scatter!(p1, apprentissage.t, apprentissage.y; ms=2, label="Apprentissage 3 %")
    plot!(p1, temps, courbes[1].gomp; label="Sans traitement", color=:gray)
    plot!(p1, temps, courbes[1].param; label="Traitement parametrique", lw=2)
    plot!(p1, temps, courbes[1].ude[1].N; label="UDE seed 11", lw=2)


    p2 = plot(; xlabel="Iteration L-BFGS", ylabel="MSE de donnees", yscale=:log10)
    for (graine, res) in zip(graines, reseaux)

        h = reduce(hcat, res.historique)

        plot!(p2, h[1,:], h[2,:]; label="Train $graine")
        plot!(p2, h[1,:], h[3,:]; label="Validation $graine", linestyle=:dash)

    end

    savefig(plot(p1,p2; layout=(2,1), size=(900,850)),
            joinpath(dossier, "02_apprentissage.png"))

    panneaux = []
    for i in 3:4

        d, c = donnees[i], courbes[i]

        p = plot(d.t, d.reference; color=:black, linestyle=:dash, lw=2,
                 label="Reference", title=noms[i], xlabel="Temps (jours)",
                 ylabel="Volume normalise", legend=:outerright, legendfontsize=12,
                 tickfontsize=12, guidefontsize=14, left_margin=8Plots.mm)
        plot!(p, d.t, c.param; label="Parametrique", lw=2)

        for j in eachindex(graines)

            plot!(p, d.t, c.ude[j].N; label="UDE $(graines[j])", lw=1.5)

        end

        push!(panneaux, p)

    end

    savefig(plot(panneaux...; layout=(1,2), size=(1000,550)), joinpath(dossier, "03_transfert.png"))

    C = [concentration(t, apprentissage.dates, k) for t in temps]
    p = plot(temps, 2.5 .* C .* apprentissage.reference; label="Correction vraie",
             color=:black, linestyle=:dash, lw=2, xlabel="Temps (jours)",
             ylabel="Correction (volume normalise / jour)",
             title="Correction sur les etats de reference", size=(850,450))

    for (graine, res) in zip(graines, reseaux)

        effet = correction.(Ref(res.p), apprentissage.reference, C, K)
        plot!(p, temps, effet; label="UDE $graine")

    end

    savefig(p, joinpath(dossier, "04_correction.png"))

end

main()
