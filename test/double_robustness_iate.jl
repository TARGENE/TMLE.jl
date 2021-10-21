module TestInteractionATE

include("helper_fns.jl")

using Test
using TMLE
using MLJ
using Distributions
using Random
using StableRNGs
using Tables
using StatsBase

mutable struct InteractionTransformer <: Static end
    
function MLJ.transform(a::InteractionTransformer, _, X)
    Xmatrix = MLJ.matrix(X)
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
    return MLJ.table(hcat(Xmatrix, Xinteracts))
end


LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity = 0

cont_interacter = @pipeline InteractionTransformer LinearRegressor name="ContInteracter"
cat_interacter = @pipeline InteractionTransformer LogisticClassifier name="CatInteracter"
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
    W = MLJ.table(float(W))
    T = (T₁ = categorical(T₁), T₂ = categorical(T₂))
    y = categorical(y)

    # Compute the theoretical ATE
    Wcomb = [1 1 1;
            1 1 0;
            1 0 0;
            1 0 1;
            0 1 0;
            0 0 0;
            0 0 1;
            0 1 1]
    ATE = 0
    for i in 1:8
        w = reshape(Wcomb[i, :], 1, 3)
        temp = μy_fn(w, [1], [1])[1]
        temp += μy_fn(w, [0], [0])[1]
        temp -= μy_fn(w, [1], [0])[1]
        temp -= μy_fn(w, [0], [1])[1]
        ATE += temp*0.5*0.5*0.5
    end
    return T, W, y, ATE
end


function binary_target_categorical_treatment_pb(rng;n=100)
    function μy_fn(W, T, Hmach)
        Thot = transform(Hmach, T)
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


    # Compute the theoretical ATE for the query
    # (CC, AT) against (CG, AA)
    Wcomb = [1 1 1;
            1 1 0;
            1 0 0;
            1 0 1;
            0 1 0;
            0 0 0;
            0 0 1;
            0 1 1]
    ATE = 0
    levels₁ = levels(T.T₁)
    levels₂ = levels(T.T₂)
    for i in 1:8
        w = reshape(Wcomb[i, :], 1, 3)
        temp = μy_fn(w, (T₁=categorical(["CC"], levels=levels₁), T₂=categorical(["AT"], levels=levels₂)), Hmach)[1]
        temp += μy_fn(w, (T₁=categorical(["CG"], levels=levels₁), T₂=categorical(["AA"], levels=levels₂)), Hmach)[1]
        temp -= μy_fn(w, (T₁=categorical(["CC"], levels=levels₁), T₂=categorical(["AA"], levels=levels₂)), Hmach)[1]
        temp -= μy_fn(w, (T₁=categorical(["CG"], levels=levels₁), T₂=categorical(["AT"], levels=levels₂)), Hmach)[1]
        ATE += temp*0.5*0.5*0.5
    end
    return T, MLJ.table(float(W)), categorical(y), ATE
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
    W = MLJ.table(float(W))
    T = (T₁ = categorical(T₁), T₂ = categorical(T₂))

    # Compute the theoretical ATE
    Wcomb = [1 1 1;
            1 1 0;
            1 0 0;
            1 0 1;
            0 1 0;
            0 0 0;
            0 0 1;
            0 1 1]
    ATE = 0
    for i in 1:8
        w = reshape(Wcomb[i, :], 1, 3)
        temp = μy_fn(w, [1], [1])[1]
        temp += μy_fn(w, [0], [0])[1]
        temp -= μy_fn(w, [1], [0])[1]
        temp -= μy_fn(w, [0], [1])[1]
        ATE += temp*0.5*0.5*0.5
    end
    return T, W, y, ATE
end


@testset "Test machine API" begin
    query = (T₁=[true, false], T₂ = [true, false])
    tmle = TMLEstimator(
            LinearRegressor(),
            FullCategoricalJoint(LogisticClassifier()),
            ContinuousFluctuation(query=query)
            )
    T, W, y, _ = continuous_target_binary_treatment_pb(StableRNG(123);n=100)
    # Fit with the machine
    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)
    # Fit using basic API
    fitresult, _, _ = TMLE.fit(tmle, 0, T, W, y)
    @test fitresult.estimate == mach.fitresult.estimate
    @test fitresult.stderror == mach.fitresult.stderror
    @test fitresult.mean_inf_curve == mach.fitresult.mean_inf_curve
end


@testset "Test Double Robustness IATE on binary_target_binary_treatment_pb" begin
    # When Q̅ is misspecified but G is well specified
    query = (T₁=[true, false], T₂=[true, false])
    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(LogisticClassifier())
    F = BinaryFluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)
    

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            binary_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [69, 14, 5, 1.5])
    @test all(abs_vars .< [0.05, 0.002, 0.0003, 3.5e-5])

    # When Q̅ is well specified  but G is misspecified
    query = (T₁=[true, false], T₂=[true, false])
    Q̅ = cat_interacter
    G = FullCategoricalJoint(ConstantClassifier())
    F = BinaryFluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)
    
    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            binary_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [53, 13, 4.5, 1.4])
    @test all(abs_vars .< [0.03, 0.002, 0.0003, 2.7e-5])

end

@testset "Test Double Robustness IATE on continuous_target_binary_treatment_pb" begin
    # When Q̅ is misspecified but G is well specified
    
    # If the order of the variables in the query do no match
    # the order in T it shouldn't matter
    query = (T₂=[true, false], T₁=[true, false])
    Q̅ = MLJ.DeterministicConstantRegressor()
    G = FullCategoricalJoint(LogisticClassifier())
    F = ContinuousFluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            continuous_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [18, 1.6, 0.5, 0.2])
    @test all(abs_vars .< [0.08, 0.002, 0.0002, 1.5e-5])

    # When Q̅ is well specified  but G is misspecified
    query = (T₁=[true, false], T₂=[true, false])
    Q̅ = cont_interacter
    G = FullCategoricalJoint(ConstantClassifier())
    F = ContinuousFluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)
    
    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            continuous_target_binary_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [2.3, 0.6, 0.3, 0.06])
    @test all(abs_vars .< [0.003, 0.0003, 2.5e-5, 2e-6])

end


@testset "Test Double Robustness IATE on binary_target_categorical_treatment_pb" begin
    # When Q̅ is misspecified but G is well specified
    query = (T₁=["CC", "CG"], T₂=["AT", "AA"])
    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(LogisticClassifier())
    F = BinaryFluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            binary_target_categorical_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [31, 8, 2, 0.6])
    @test all(abs_vars .< [0.03, 0.006, 0.0004, 2.7e-5])

    # When Q̅ is well specified but G is misspecified
    query = (T₁=["CC", "CG"], T₂=["AT", "AA"])
    Q̅ = cat_interacter
    G = FullCategoricalJoint(ConstantClassifier())
    F = BinaryFluctuation(query=query)
    tmle = TMLEstimator(Q̅, G, F)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            binary_target_categorical_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [22, 7, 2.1, 0.7])
    @test all(abs_vars .< [0.04, 0.005, 0.0004, 3.5e-5])
end


end;


true