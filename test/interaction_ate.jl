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


function binary_target_binary_treatment_pb(rng;n=100)
    μy_fn(W, T₁, T₂) = TMLE.expit(2W[:, 1] .+ 1W[:, 2] .- 2W[:, 3] .- T₁ .+ T₂ .+ 2*T₁ .* T₂)
    # Sampling W: Bernoulli
    W = rand(rng, Bernoulli(0.5), n, 3)

    # Sampling T₁, T₂ from W: Softmax
    θ = [1 2 -3 -2; 
         -2 4 6 0 ;
         3 -1 -4 2]
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
    θ = [1 2 -3 -2; 
         -2 4 6 0 ;
         3 -1 -4 2]
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
    tmle = InteractionATEEstimator(
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


# Here I illustrate the Double Robust behavior by
# misspecifying one of the models and the TMLE still converges
cont_interacter = @pipeline InteractionTransformer LinearRegressor name="ContInteracter"
cat_interacter = @pipeline InteractionTransformer LogisticClassifier name="CatInteracter"
query = (T₁=[true, false], T₂ = [true, false])
grid = (
    (problem=continuous_target_binary_treatment_pb, 
    fluctuation=ContinuousFluctuation(query=query), 
    subgrid=((cont_interacter, FullCategoricalJoint(ConstantClassifier()), [3.7, 1.6, 0.46, 0.1], [0.009, 0.002, 8.5e-5, 5.9e-6]),
             (MLJ.DeterministicConstantRegressor(), FullCategoricalJoint(LogisticClassifier()), [118, 58, 18, 8.7], [7.7, 2.5, 0.16, 0.009]))
    ),
    (problem=binary_target_binary_treatment_pb, 
    fluctuation=BinaryFluctuation(query=query), 
    subgrid=((cat_interacter, FullCategoricalJoint(ConstantClassifier()), [80, 37, 10, 1.5], [0.02, 0.004, 0.0009, 3.6e-5]),
            (ConstantClassifier(), FullCategoricalJoint(LogisticClassifier()), [167, 79, 33, 14], [0.27, 0.095, 0.017, 0.002]))
    )
)
rng = StableRNG(1234)
Ns = [100, 1000, 10000, 100000]
@testset "IATE TMLE Double Robustness on $(replace(string(problem_set.problem), '_' => ' '))
            - E[Y|W,T] is a $(string(typeof(y_model)))
            - p(T|W) is a $(string(typeof(t_model)))" (
    for problem_set in grid, (y_model, t_model, expected_bias_upb, expected_var_upb) in problem_set.subgrid
        tmle = InteractionATEEstimator(
            y_model,
            t_model,
            problem_set.fluctuation
            )
        abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            problem_set.problem,
            rng,
            Ns
            )
        println("--Next--")
        println(abs_mean_rel_errors)
        println(expected_bias_upb)
        # Check the bias and variance are converging to 0
        # @test all(abs_mean_rel_errors .< expected_bias_upb)
        # @test all(abs_vars .< expected_var_upb)
end);

@testset "Test TMLE on binary_target_categorical_treatment_pb" begin
    # (CC, AT) against (CG, AA)
    query = (T₁=categorical(["CC", "CG"], levels=levels(T.T₁)), 
             T₂=categorical(["AT", "AA"], levels=levels(T.T₂)))
    Q̅ = ConstantClassifier()
    G = FullCategoricalJoint(LogisticClassifier())
    F = BinaryFluctuation(query=query)
    tmle = InteractionATEEstimator(Q̅, G, F)

    abs_mean_rel_errors, abs_vars = asymptotics(
            tmle,                                 
            binary_target_categorical_treatment_pb,
            StableRNG(123),
            Ns
            )
    @test all(abs_mean_rel_errors .< [31, 8, 2, 0.6])
    @test all(abs_vars .< [0.03, 0.006, 0.0004, 2.7e-5])
end


end


true