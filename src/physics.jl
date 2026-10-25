using OrdinaryDiffEqVerner: Vern7
using SciMLBase: ODEProblem, solve, successful_retcode

gompertz(N, r, K) = r * N * log(K/N)

function concentration(t, dates, k; gauche=false)

    C = zero(t)

    for date in dates

        if t >= date && (!gauche || t > date || date == 0)
            C += exp(-k * (t - date))

        end
    end

    return C

end

function verifier_physique(r, K, N0, k, dates)

    all(isfinite, (r, K, N0, k)) && r >= 0 && k >= 0 && K > 0 && N0 > 0 || throw(ArgumentError("r,k >= 0 et N0,K > 0"))
    all(d -> isfinite(d) && d >= 0, dates) || throw(ArgumentError("date injection : finie & =>0"))

end

# expm1(x)/x, avec sa limite et un developpement stable au voisinage de zero
function exprel(x)

    abs(x) < 1e-5 && return 1 + x/2 + x^2/6 + x^3/24 + x^4/120

    return expm1(x) / x

end

function reference_analytique(temps, dates, r, K, delta; N0=0.1, k=0.25)

    verifier_physique(r, K, N0, k, dates)
    isfinite(delta) && delta >= 0 || throw(ArgumentError("delta doit etre fini et >= 0"))
    all(t -> isfinite(t) && t >= 0, temps) || throw(ArgumentError("temps finis et >= 0"))

    N = map(temps) do t
        z = log(N0 / K) * exp(-r * t)
        for date in dates

            if t >= date
                s = t - date
                phi = r >= k ? s * exp(-k*s) * exprel(-(r-k)*s) : s * exp(-r*s) * exprel(-(k-r)*s) # Integrale de exp(-r*(t-u))*exp(-k*(u-date)) du
                z -= delta * phi 

            end
        end

        K * exp(z)

    end

    all(n -> isfinite(n) && n > 0, N) || error("Ref non fini || non fini")

    return N

end

function integrer_log(champ!, p, temps, dates; N0=0.1, K=1.0, abstol=1e-10, reltol=1e-8)

    t = collect(Float64, temps)
    length(t) >= 2 && all(isfinite, t) && first(t) == 0 && all(diff(t) .> 0) || throw(ArgumentError("temps >0 et croiss "))
    N0 > 0 && K > 0 || throw(ArgumentError("N0 & K >0"))

    z0 = [log(N0/K) + zero(first(p))]
    prob = ODEProblem(champ!, z0, (first(t), last(t)), p)
    injections = sort(unique(filter(d -> first(t) < d < last(t), dates)))

    sol = solve(prob, Vern7(); saveat=t, abstol, reltol, tstops=injections, d_discontinuities=injections)
    successful_retcode(sol) || error("Integration stop : $(sol.retcode)")
    sol.t == t || error("Solution incomplète : manque des temps")

    N = K .* exp.(vec(Array(sol)))
    all(isfinite, N) && all(N .> 0) || error("Volume <0 || pas fini")

    return (N=N, statut=string(sol.retcode), pas=sol.destats.naccept)

end

function simuler_parametrique(delta, temps, dates, r, K; N0=0.1, k=0.25, abstol=1e-10, reltol=1e-8)

    verifier_physique(r, K, N0, k, dates)
    isfinite(delta) && delta >= 0 || throw(ArgumentError("delta fini et >= 0"))

    function champ!(dz, z, p, t)

        dz[1] = -r * z[1] - p[1] * concentration(t, dates, k; gauche=true)

    end

    return integrer_log(champ!, [delta], temps, dates; N0, K, abstol, reltol)
    
end
