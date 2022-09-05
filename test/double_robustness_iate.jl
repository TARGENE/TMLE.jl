module TestInteractionATE

include("helper_fns.jl")

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

mutable struct InteractionTransformer <: Static end
    
function MLJBase.transform(a::InteractionTransformer, _, X)
    Xmatrix = MLJBase.matrix(X)
    nrows, ncols = size(Xmatrix)
    ninter = Int(ncols*(ncols-1)/2)
    Xinteracts = Matrix{Float64}(undef, nrows, ninter)
    i = 0
    for col₁ in 1:(ncols-1)
        for col₂ in (col₁+1):ncols
            i += 1
            Xinteracts[:, i] = Xmatrix[:, col₁] .* Xmatrix[:, col₂]
        end
    end
    return MLJBase.table(hcat(Xmatrix, Xinteracts))
end

cont_interacter = InteractionTransformer |> LinearRegressor
cat_interacter = InteractionTransformer |> LogisticClassifier
Ns = [100, 1000, 10000, 100000]


function binary_target_binary_treatment_pb(rng;n=100)
    μy_fn(W, T₁, T₂) = TMLE.expit(2W[:, 1] .+ 1W[:, 2] .- 2W[:, 3] .- T₁ .+ T₂ .+ 2*T₁ .* T₂)
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
    y = categorical(y)

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
    return (T₁=T₁, T₂=T₂, W₁=W[:, 1], W₂=W[:, 2], W₃=W[:, 3], y=y), IATE
end


function binary_target_categorical_treatment_pb(rng;n=100)
    function μy_fn(W, T, Hmach)
        Thot = MLJBase.transform(Hmach, T)
        TMLE.expit(2W[:, 1] .+ 1W[:, 2] .- 2W[:, 3] 
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
    W = rand(rng, Bernoulli(0.5), n, 3)

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
    return (T₁=T.T₁, T₂=T.T₂, W₁=W[:, 1], W₂=W[:, 2], W₃=W[:, 3], y=categorical(y)), IATE
end


function continuous_target_binary_treatment_pb(rng;n=100)
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

    return (T₁=T₁, T₂=T₂,  W₁=W[:, 1], W₂=W[:, 2], W₃=W[:, 3], y=y), IATE
end

@testset "Test Double Robustness IATE on binary_target_binary_treatment_pb" begin
    Ψ = IATE(
        target=:y,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
        confounders = [:W₁, :W₂, :W₃]
    )
    # When Q is misspecified but G is well specified
    η_spec = (
        Q = ConstantClassifier(),
        G = LogisticClassifier(lambda=0)
    )
    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        binary_target_binary_treatment_pb, 
        StableRNG(123), 
        Ns)
    @test all_tmle_better_than_initial(tmle_results, initial_results, Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results, tol=1e-7) 

    # When Q is well specified  but G is misspecified
    η_spec = (
        Q = cat_interacter,
        G = ConstantClassifier()
    )
    
    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        binary_target_binary_treatment_pb, 
        StableRNG(123), 
        Ns)
    # 3/4 are better
    @test sum(abserrors(tmle_results, Ψ₀) .< abserrors(initial_results, Ψ₀)) == 3
    @test first_better_than_last(tmle_results, Ψ₀)
    # This is quite far away...
    @test tolerance(tmle_results[end], Ψ₀, 0.5)
    @test all_solves_ice(tmle_results, tol=1e-8)

end

@testset "Test Double Robustness IATE on continuous_target_binary_treatment_pb" begin
    Ψ = IATE(
        target=:y,
        treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
        confounders = [:W₁, :W₂, :W₃]
    )
    # When Q is misspecified but G is well specified
    η_spec = (
        Q = MLJModels.DeterministicConstantRegressor(),
        G = LogisticClassifier(lambda=0)
    )

    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        continuous_target_binary_treatment_pb, 
        StableRNG(123), 
        Ns)

    @test all_tmle_better_than_initial(tmle_results, initial_results, Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results, tol=1e-7) 

    # When Q is well specified  but G is misspecified
    η_spec = (
        Q = cont_interacter,
        G = ConstantClassifier()
    )

    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        continuous_target_binary_treatment_pb, 
        StableRNG(123), 
        Ns)

    @test sum(abserrors(tmle_results, Ψ₀) .< abserrors(initial_results, Ψ₀)) == 3
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results, tol=1e-7) 
end


@testset "Test Double Robustness IATE on binary_target_categorical_treatment_pb" begin
    Ψ = IATE(
        target=:y,
        treatment=(T₁=(case="CC", control="CG"), T₂=(case="AT", control="AA")),
        confounders = [:W₁, :W₂, :W₃]
    )
    # When Q is misspecified but G is well specified
    η_spec = (
        Q = ConstantClassifier(),
        G = LogisticClassifier(lambda=0)
    )

    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        binary_target_categorical_treatment_pb, 
        StableRNG(123), 
        Ns)

    @test all_tmle_better_than_initial(tmle_results, initial_results, Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results, tol=1e-7)

    # When Q is well specified but G is misspecified
    η_spec = (
        Q = cat_interacter,
        G = ConstantClassifier()
    )

    tmle_results, initial_results, Ψ₀ = asymptotics(
        Ψ, 
        η_spec, 
        binary_target_categorical_treatment_pb, 
        StableRNG(123), 
        Ns)

    @test all_tmle_better_than_initial(tmle_results, initial_results, Ψ₀)
    @test first_better_than_last(tmle_results, Ψ₀)
    @test tolerance(tmle_results[end], Ψ₀, 0.011)
    @test all_solves_ice(tmle_results, tol=1e-7)
end


end;


true