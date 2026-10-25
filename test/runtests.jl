using Test, ForwardDiff, LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "src", "physics.jl"))
include(joinpath(@__DIR__, "..", "src", "models.jl"))
include(joinpath(@__DIR__, "..", "src", "tools.jl"))

@testset "Gompertz : solution et equilibre" begin
    @test gompertz(1.0, 0.35, 1.0) == 0
    @test gompertz(0.5, 0.35, 1.0) > 0
    @test gompertz(1.2, 0.35, 1.0) < 0
    t = 0.0:0.5:20.0
    for N0 in [0.1, 1.0, 1.2]
        exact = reference_analytique(t, Float64[], 0.35, 1.0, 0.0; N0)
        sol = simuler_parametrique(0.0, t, Float64[], 0.35, 1.0; N0)
        @test maximum(abs, sol.N .- exact) < 1e-7
    end
end

@testset "Traitement : injections et limite r=k" begin
    t, dates = 0.0:0.5:20.0, [3.25, 7.1, 11.0]
    for r in [0.35, 0.25, 0.25-1e-9, 0.25+1e-9]
        exact = reference_analytique(t, dates, r, 1.0, 2.5)
        sol = simuler_parametrique(2.5, t, dates, r, 1.0)
        erreur = maximum(abs, sol.N .- exact)
        @printf("Reference traitee r=%.10f : erreur max %.3e\n", r, erreur)
        @test erreur < 1e-7
    end
end

@testset "UDE : contraintes et coefficient constant" begin
    p = initialiser_reseau(11)
    @test length(p) == 33
    @test correction(p, 0.5, 0.0, 1.0) == 0
    @test isfinite(softplus(1000.0))
    @test ForwardDiff.derivative(softplus, 0.0) ≈ 0.5
    # Le reseau constant doit reproduire le modele parametrique.
    theta = zeros(33)
    theta[33] = log(expm1(2.5))
    t, dates = 0.0:0.5:60.0, [10.0, 20.0, 30.0]
    sol = simuler_ude(theta, t, dates, 0.35, 1.0)
    exact = reference_analytique(t, dates, 0.35, 1.0, 2.5)
    @test maximum(abs, sol.N .- exact) < 1e-7
    @test all(0 .< sol.N .<= 1.0+1e-8)
end

@testset "Gradient : controle directionnel" begin
    d = creer_donnees(0.0:1.0:20.0, [5.0, 10.0], 42)
    p = initialiser_reseau(11)
    v = randn(MersenneTwister(123), length(p)); v /= norm(v)
    f(q) = perte_ude(q, d, 0.35, 1.0; abstol=1e-12, reltol=1e-10)
    derivee = dot(ForwardDiff.gradient(f, p), v)
    @test abs(derivee) > 1e-6
    erreurs = Float64[]
    for h in [1e-3, 1e-4, 1e-5]
        df = (f(p+h*v) - f(p-h*v)) / (2h)
        erreur = abs(df-derivee) / abs(derivee)
        push!(erreurs, erreur)
        @printf("Gradient : h=%.0e AD=%.8e DF=%.8e erreur relative=%.3e\n",
                h, derivee, df, erreur)
    end
    @test minimum(erreurs) < 1e-3
end

@testset "Sensibilite aux tolerances" begin
    t, dates = 0.0:0.5:60.0, [10.0, 20.0, 30.0]
    p = initialiser_reseau(11)
    usuel = simuler_ude(p, t, dates, 0.35, 1.0)
    fin = simuler_ude(p, t, dates, 0.35, 1.0; abstol=1e-12, reltol=1e-10)
    ecart = maximum(abs, usuel.N .- fin.N)
    @printf("UDE : variation avec tolerances plus fines %.3e\n", ecart)
    @test ecart < 1e-6
end

@testset "Reference stable et derivee pour r=k" begin
    # L'ancienne expression formait 0*Inf a l'instant final de ce cas.
    ref = reference_analytique([0.,1000.], [0.], .01, 1., 2.5; k=1.)
    exact = setprecision(256) do
        r, k, t, d = BigFloat.((.01,1.,1000.,2.5))
        z = log(BigFloat(.1))*exp(-r*t) - d*(exp(-k*t)-exp(-r*t))/(r-k)
        Float64(exp(z))
    end
    @test ref[end] ≈ exact rtol=1e-14
    @test all(isfinite, ref)
    t, date, r, delta = 20., 3.25, .25, 2.5
    s = t-date
    N = reference_analytique([t],[date],r,1.,delta)[1]
    derivee = N*(-t*log(.1)*exp(-r*t) + delta*s^2*exp(-r*s)/2)
    ad = ForwardDiff.derivative(q->reference_analytique([t],[date],q,1.,delta)[1], r)
    @test ad ≈ derivee rtol=1e-12
end

@testset "Injections aux bornes et valeurs laterales" begin
    @test concentration(0.,[0.],.25;gauche=true) == 1.
    @test concentration(2.,[2.],.25) == 1.
    @test concentration(2.,[2.],.25;gauche=true) == 0.
    @test concentration(nextfloat(2.),[2.],.25;gauche=true) ≈ 1.
    temps = 0.:.25:20.
    for dates in [[0.,5.3],[20.],[7.1,3.25,3.25]]
        ref = reference_analytique(temps,dates,.35,1.,2.5)
        sol = simuler_parametrique(2.5,temps,dates,.35,1.)
        @test maximum(abs,sol.N.-ref) < 1e-7
        @test sol.N[1] ≈ .1
    end
end

@testset "Entrees invalides et dimensions" begin
    p = initialiser_reseau(11)
    for dates in [[NaN],[-1.],[Inf]]
        @test_throws ArgumentError reference_analytique([0.,1.],dates,.35,1.,2.5)
        @test_throws ArgumentError simuler_ude(p,[0.,1.],dates,.35,1.)
    end
    for n in [32,34]
        @test_throws ArgumentError reseau(zeros(n),.5,1.)
        @test_throws ArgumentError simuler_ude(zeros(n),[0.,1.],[0.],.35,1.)
    end
    @test_throws ArgumentError simuler_ude(p,[0.,Inf],[0.],.35,1.)
    @test_throws ArgumentError simuler_parametrique(-1.,[0.,1.],[0.],.35,1.)
    @test_throws ArgumentError reference_analytique([0.,1.],[0.],.35,1.,2.5;k=-.25)
end

@testset "Condition initiale et elimination transmises aux pertes" begin
    d = creer_donnees(0.:.5:30.,[4.,12.],52;bruit=0.,N0=.2,k=.4)
    p = zeros(33);p[33] = log(expm1(2.5))
    @test perte_ude(p,d,.35,1.;alpha=0.) < 1e-14
    fit = calibrer_traitement(d,.35,1.)
    @test fit.delta ≈ 2.5 rtol=1e-6
    @test Optim.converged(fit.resultat)
end
