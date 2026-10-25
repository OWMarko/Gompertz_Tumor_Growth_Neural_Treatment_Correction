using Random, Statistics, TOML, Optim, ForwardDiff

function creer_donnees(temps, dates, graine; bruit=0.03, N0=0.1, k=0.25)

    # Verite synthetique est utilisee pour generer et evaluer

    t = collect(Float64, temps)
    N = reference_analytique(t, dates, 0.35, 1.0, 2.5; N0, k)
    y = N .* (1 .+ bruit .* randn(MersenneTwister(graine), length(t)))

    return (t=t, dates=Float64.(dates), y=y, reference=N, N0=N0, k=k)

end

function ecrire_csv(chemin, entete, lignes)

    open(chemin, "w") do io
        println(io, join(entete, ','))

        for ligne in lignes

            println(io, join(ligne, ','))

        end
    end
end

function calibrer_croissance(donnees)

    function perte(q)
        r, K = exp.(q)
        N = reference_analytique(donnees.t, Float64[], r, K, 0.0; N0=donnees.N0, k=donnees.k)

        return mean(abs2, N .- donnees.y)

    end

    res = Optim.optimize(perte, log.([0.2, 0.8]), Optim.LBFGS(), Optim.Options(iterations=200, g_tol=1e-10); autodiff=:forward)
    r, K = exp.(Optim.minimizer(res))

    return (r=r, K=K, resultat=res)

end

function calibrer_traitement(donnees, r, K)

    function perte(q)

        N = reference_analytique(donnees.t, donnees.dates, r, K, exp(q[1]); N0=donnees.N0, k=donnees.k)

        return mean(abs2, N .- donnees.y)

    end

    res = Optim.optimize(perte, [0.0], Optim.LBFGS(), Optim.Options(iterations=100, g_tol=1e-10); autodiff=:forward)

    return (delta=exp(Optim.minimizer(res)[1]), resultat=res)

end

function entrainer_ude(p0, apprentissage, validation, r, K; alpha=1e-6, iterations=200)

    perte(p) = perte_ude(p, apprentissage, r, K; alpha)

    res = Optim.optimize(perte, copy(p0), Optim.LBFGS(), Optim.Options(iterations=iterations, g_tol=1e-8, store_trace=true, extended_trace=true); autodiff=:forward)

    meilleur_p = copy(p0)
    meilleure_validation = Inf
    historique = Vector{Vector{Float64}}()
    meilleur_iteration = 0

    for etat in Optim.trace(res)

        p = etat.metadata["x"]
        mse_train = perte_ude(p, apprentissage, r, K; alpha=0.0)
        mse_val = perte_ude(p, validation, r, K; alpha=0.0)
        push!(historique, [etat.iteration, mse_train, mse_val, etat.value])

        if mse_val < meilleure_validation
            meilleure_validation = mse_val
            meilleur_p = copy(p)
            meilleur_iteration = etat.iteration
        end
    end

    return (p=meilleur_p, historique=historique, iteration=meilleur_iteration, iterations=Optim.iterations(res), converge=Optim.converged(res))

end

function scores(N, donnees)

    return (mse_observations=mean(abs2, N .- donnees.y), rmse_reference=sqrt(mean(abs2, N .- donnees.reference)), rmse_log=sqrt(mean(abs2, log.(N ./ donnees.reference))))

end
