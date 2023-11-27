module TestInteractionATE

using Test
using TMLE
using MLJBase
using Distributions
using Random
using StableRNGs
using Tables
using StatsBase
using MLJModels
using MLJLinearModels
using LogExpFunctions

include(joinpath(pkgdir(TMLE), "test", "helper_fns.jl"))

cont_interacter = InteractionTransformer(order=2) |> LinearRegressor
cat_interacter = InteractionTransformer(order=2) |> LogisticClassifier(lambda=1.)


function binary_outcome_binary_treatment_pb(;n=100)
    rng = StableRNG(123)
    μy_fn(W, T₁, T₂) = logistic.(2W[:, 1] .+ 1W[:, 2] .- 2W[:, 3] .- T₁ .+ T₂ .+ 2*T₁ .* T₂)
    # Sampling W: Bernoulli
    W = rand(rng, Bernoulli(0.5), n, 3)

    # Sampling T₁, T₂ from W: Softmax
    θ = rand(rng, 3, 4)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    T = [sample(rng, [1, 2, 3, 4], Weights(softmax[i, :])) for i in 1:n]
    T₁ = [t in (1,2) ? true : false for t in T]
    T₂ = [t in (1,3) ? true : false for t in T]

    # Sampling y from T₁, T₂, W: Logistic
    μy = μy_fn(W, T₁, T₂)
    y = [rand(rng, Bernoulli(μy[i])) for i in 1:n]

    # Respect the Tables.jl interface and convert types
    W = float(W)
    T₁ = categorical(T₁)
    T₂ = categorical(T₂)
    Y = categorical(y)
    dataset = (T₁=T₁, T₂=T₂, W₁=W[:, 1], W₂=W[:, 2], W₃=W[:, 3], Y=Y)
    # Compute the theoretical IATE
    Wcomb = [1 1 1;
            1 1 0;
            1 0 0;
            1 0 1;
            0 1 0;
            0 0 0;
            0 0 1;
            0 1 1]
    IATE = 0
    for i in 1:8
        w = reshape(Wcomb[i, :], 1, 3)
        temp = μy_fn(w, [1], [1])[1]
        temp += μy_fn(w, [0], [0])[1]
        temp -= μy_fn(w, [1], [0])[1]
        temp -= μy_fn(w, [0], [1])[1]
        IATE += temp*0.5*0.5*0.5
    end
    return dataset, IATE
end


function binary_outcome_categorical_treatment_pb(;n=100)
    rng = StableRNG(123)
    function μy_fn(W, T, Hmach)
        Thot = MLJBase.transform(Hmach, T)
        logistic.(2W[:, 1] .+ 1W[:, 2] .- 2W[:, 3] 
                    .- Thot[1] .+ Thot[2] .+ 2Thot[3] .- 3Thot[4]
                    .+ 2*Thot[1].*Thot[2]
                    .+ 1*Thot[1].*Thot[3]
                    .- 4*Thot[1].*Thot[4]
                    .- 3*Thot[2].*Thot[3]
                    .+ 1.5*Thot[2].*Thot[4]
                    .- 2.5*Thot[3].*Thot[4]
                    )
    end
    # Sampling W:
    W = float(rand(rng, Bernoulli(0.5), n, 3))

    # Sampling T from W:
    # T₁, T₂ will have 3 categories each
    # This is embodied by a 9 dimensional full joint
    θ = rand(rng, 3, 9)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    encoding = collect(Iterators.product(["CC", "GG", "CG"], ["TT", "AA", "AT"]))
    T = [sample(rng, encoding, Weights(softmax[i, :])) for i in 1:n]
    T = (T₁=categorical([t[1] for t in T]), T₂=categorical([t[2] for t in T]))

    Hmach = machine(OneHotEncoder(drop_last=true), T)
    fit!(Hmach, verbosity=0)

    # Sampling y from T, W:
    μy = μy_fn(W, T, Hmach)
    y = [rand(rng, Bernoulli(μy[i])) for i in 1:n]
    dataset = (T₁=T.T₁, T₂=T.T₂, W₁=W[:, 1], W₂=W[:, 2], W₃=W[:, 3], Y=categorical(y))
    # Compute the theoretical IATE for the query
    # (CC, AT) against (CG, AA)
    Wcomb = [1 1 1;
            1 1 0;
            1 0 0;
            1 0 1;
            0 1 0;
            0 0 0;
            0 0 1;
            0 1 1]
            IATE = 0
    levels₁ = levels(T.T₁)
    levels₂ = levels(T.T₂)
    for i in 1:8
        w = reshape(Wcomb[i, :], 1, 3)
        temp = μy_fn(w, (T₁=categorical(["CC"], levels=levels₁), T₂=categorical(["AT"], levels=levels₂)), Hmach)[1]
        temp += μy_fn(w, (T₁=categorical(["CG"], levels=levels₁), T₂=categorical(["AA"], levels=levels₂)), Hmach)[1]
        temp -= μy_fn(w, (T₁=categorical(["CC"], levels=levels₁), T₂=categorical(["AA"], levels=levels₂)), Hmach)[1]
        temp -= μy_fn(w, (T₁=categorical(["CG"], levels=levels₁), T₂=categorical(["AT"], levels=levels₂)), Hmach)[1]
        IATE += temp*0.5*0.5*0.5
    end
    return dataset, IATE
end


function continuous_outcome_binary_treatment_pb(;n=100)
    rng = StableRNG(123)
    μy_fn(W, T₁, T₂) = 2W[:, 1] .+ 1W[:, 2] .- 2W[:, 3] .- T₁ .+ T₂ .+ 2*T₁ .* T₂
    # Sampling W: Bernoulli
    W = rand(rng, Bernoulli(0.5), n, 3)

    # Sampling T₁, T₂ from W: Softmax
    θ = rand(rng, 3, 4)
    softmax = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    T = [sample(rng, [1, 2, 3, 4], Weights(softmax[i, :])) for i in 1:n]
    T₁ = [t in (1,2) ? true : false for t in T]
    T₂ = [t in (1,3) ? true : false for t in T]

    # Sampling y from T₁, T₂, W: Logistic
    μy = μy_fn(W, T₁, T₂)
    y = μy + rand(rng, Normal(0, 0.1), n)

    # Respect the Tables.jl interface and convert types
    W = float(W)
    T₁ = categorical(T₁)
    T₂ = categorical(T₂)

    dataset = (T₁=T₁, T₂=T₂,  W₁=W[:, 1], W₂=W[:, 2], W₃=W[:, 3], Y=y)
    # Compute the theoretical ATE
    Wcomb = [1 1 1;
            1 1 0;
            1 0 0;
            1 0 1;
            0 1 0;
            0 0 0;
            0 0 1;
            0 1 1]
    IATE = 0
    for i in 1:8
        w = reshape(Wcomb[i, :], 1, 3)
        temp = μy_fn(w, [1], [1])[1]
        temp += μy_fn(w, [0], [0])[1]
        temp -= μy_fn(w, [1], [0])[1]
        temp -= μy_fn(w, [0], [1])[1]
        IATE += temp*0.5*0.5*0.5
    end
    return dataset, IATE
end

@testset "Test Double Robustness IATE on binary_outcome_binary_treatment_pb" begin
    dataset, Ψ₀ = binary_outcome_binary_treatment_pb(n=10_000)
    Ψ = IATE(
        outcome=:Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # When Q is misspecified but G is well specified
    models = (
        Y = with_encoder(ConstantClassifier()),
        T₁ = LogisticClassifier(lambda=0),
        T₂ = LogisticClassifier(lambda=0),
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-9)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-9)
    # The initial estimate is far away
    naive = NAIVE(models.Y)
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0

    # When Q is well specified  but G is misspecified
    models = (
        Y = with_encoder(LogisticClassifier(lambda=0)),
        T₁ = ConstantClassifier(),
        T₂ = ConstantClassifier(),
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-9)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-9)
    # The initial estimate is far away
    naive = NAIVE(models.Y)
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result ≈ -0.0 atol=1e-1
end

@testset "Test Double Robustness IATE on continuous_outcome_binary_treatment_pb" begin
    dataset, Ψ₀ = continuous_outcome_binary_treatment_pb(n=10_000)
    Ψ = IATE(
        outcome = :Y,
        treatment_values = (
            T₁=(case=true, control=false), 
            T₂=(case=true, control=false)
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # When Q is misspecified but G is well specified
    models = (
        Y = with_encoder(MLJModels.DeterministicConstantRegressor()),
        T₁ = LogisticClassifier(lambda=0),
        T₂ = LogisticClassifier(lambda=0),
    )

    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)
    # The initial estimate is far away
    naive = NAIVE(models.Y)
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0

    # When Q is well specified  but G is misspecified
    models = (
        Y = with_encoder(cont_interacter),
        T₁ = ConstantClassifier(),
        T₂ = ConstantClassifier(),
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-10)
end


@testset "Test Double Robustness IATE on binary_outcome_categorical_treatment_pb" begin
    dataset, Ψ₀ = binary_outcome_categorical_treatment_pb(n=30_000)
    Ψ = IATE(
        outcome=:Y,
        treatment_values= (
            T₁=(case="CC", control="CG"), 
            T₂=(case="AT", control="AA")
        ),
        treatment_confounders = (
            T₁=[:W₁, :W₂, :W₃],
            T₂=[:W₁, :W₂, :W₃],
        )
    )
    # When Q is misspecified but G is well specified
    models = (
        Y = with_encoder(ConstantClassifier()),
        T₁ = LogisticClassifier(lambda=0),
        T₂ = LogisticClassifier(lambda=0)
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-5)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-10)

    # The initial estimate is far away
    naive = NAIVE(models.Y)
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result == 0

    # When Q is well specified but G is misspecified
    models = (
        Y = with_encoder(cat_interacter),
        T₁ = ConstantClassifier(),
        T₂ = ConstantClassifier(),
    )
    dr_estimators = double_robust_estimators(models)
    results, cache = test_coverage_and_get_results(dr_estimators, Ψ, Ψ₀, dataset; verbosity=0)
    test_mean_inf_curve_almost_zero(results.tmle; atol=1e-5)
    test_mean_inf_curve_almost_zero(results.ose; atol=1e-15)

    # The initial estimate is far away
    naive = NAIVE(models.Y)
    naive_result, cache = naive(Ψ, dataset; cache=cache, verbosity=0)
    @test naive_result ≈ -0.02 atol=1e-2
end


end;


true