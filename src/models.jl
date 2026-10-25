using LinearAlgebra, Random, Statistics

# Reseau 2 -> 8 -> 1 avec 16 + 8 + 8 + 1 = 33 parametres

function reseau(p, N_reduit, C)

    length(p) == 33 || throw(ArgumentError("Le reseau attend exactement 33 parametres."))
    W1 = reshape(view(p, 1:16), 8, 2)
    b1 = view(p, 17:24)
    w2 = view(p, 25:32)
    h = tanh.(W1 * [N_reduit, C] + b1)

    return dot(w2, h) + p[33]

end

softplus(x) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

efficacite(p, N, C, K) = softplus(reseau(p, N / K, C))
correction(p, N, C, K) = N * C * efficacite(p, N, C, K)

function initialiser_reseau(graine)

    rng = MersenneTwister(graine)

    return vcat(0.2 .* vec(randn(rng, 8, 2)), zeros(8), 0.2 .* randn(rng, 8), log(expm1(1.0)))

end

function simuler_ude(p, temps, dates, r, K; N0=0.1, k=0.25, abstol=1e-10, reltol=1e-8)

    verifier_physique(r, K, N0, k, dates)
    length(p) == 33 && all(isfinite, p) || throw(ArgumentError("Para erreur"))

    function champ!(dz, z, theta, t)

        N = K * exp(z[1])
        C = concentration(t, dates, k; gauche=true)
        dz[1] = -r * z[1] - C * efficacite(theta, N, C, K)

    end

    return integrer_log(champ!, p, temps, dates; N0, K, abstol, reltol)

end

function perte_ude(p, donnees, r, K; alpha=1e-6, abstol=1e-10, reltol=1e-8)

    N = simuler_ude(p, donnees.t, donnees.dates, r, K; N0=donnees.N0, k=donnees.k, abstol, reltol).N

    return mean(abs2, N .- donnees.y) + alpha * sum(abs2, p)
    
end
