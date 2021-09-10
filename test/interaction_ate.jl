module TestInteractionATE

include("utils.jl")

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


function categorical_problem(rng;n=100)
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


function continuous_problem(rng;n=100)
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


@testset "Test Helper functions" begin
    # Test tomultivariate function
    # The mapping is fixed and harcoded
    T = (t1 = categorical([true, false, false, true, true, true, false]),
         t2 = categorical([true, true, true, true, true, false, false]))
    W = MLJ.table(rand(7, 3))
    t_target = TMLE.tomultivariate(T)
    @test t_target == categorical([1, 3, 3, 1, 1, 2, 4])
    # Test compute_covariate
    
    t_likelihood_estimate = machine(ConstantClassifier(), W, t_target)
    fit!(t_likelihood_estimate, verbosity=0)
    
    Tnames = Tables.columnnames(T)
    T = NamedTuple{Tnames}([float(Tables.getcolumn(T, colname)) for colname in Tnames])
    cov = TMLE.compute_covariate(t_likelihood_estimate, W, T, t_target)
    @test cov == [2.3333333333333335,
                 -3.5,
                 -3.5,
                 2.3333333333333335,
                 2.3333333333333335,
                 -7.0,
                 7.0]

end

@testset "Test machine API" begin
    tmle = InteractionATEEstimator(
            LinearRegressor(),
            LogisticClassifier(),
            ContinuousFluctuation()
            )
    T, W, y, _ = continuous_problem(StableRNG(123);n=100)
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
grid = (
    (problem=continuous_problem, 
    fluctuation=ContinuousFluctuation(), 
    subgrid=((cont_interacter, ConstantClassifier(), [3.7, 1.6, 0.46, 0.1], [0.009, 0.002, 8.5e-5, 5.9e-6]),
             (MLJ.DeterministicConstantRegressor(), LogisticClassifier(), [118, 58, 18, 8.7], [7.7, 2.5, 0.16, 0.009]))
    ),
    (problem=categorical_problem, 
    fluctuation=BinaryFluctuation(), 
    subgrid=((cat_interacter, ConstantClassifier(), [80, 37, 10, 1.5], [0.02, 0.004, 0.0009, 3.6e-5]),
            (ConstantClassifier(), LogisticClassifier(), [167, 79, 33, 14], [0.27, 0.095, 0.017, 0.002]))
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
        # Check the bias and variance are converging to 0
        @test all(abs_mean_rel_errors .< expected_bias_upb)
        @test all(abs_vars .< expected_var_upb)
end);

# T, W, y, ATE = continuous_problem(StableRNG(123);n=100)

# iate = InteractionATEEstimator(
#     LinearRegressor(),
#     LogisticClassifier(),
#     ContinuousFluctuation())

# mach = machine(iate, T, W, y)
# fit!(mach)


end


true