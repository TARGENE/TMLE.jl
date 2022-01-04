module TestUtils

using Test
using TMLE
using MLJ
using StableRNGs
using Distributions

LinearBinaryClassifier = @load LinearBinaryClassifier pkg=GLM verbosity=0
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

@testset "Test expected_value & maybelogit" begin
    n = 100
    X = MLJ.table(rand(n, 3))

    # Probabilistic Classifier
    y = categorical(rand([0, 1], n))
    mach = machine(ConstantClassifier(), X, y)
    fit!(mach; verbosity=0)
    proba = mach.fitresult[2][2]
    ŷ = predict(mach)
    expectation = TMLE.expected_value(ŷ, typeof(mach.model), target_scitype(mach.model))
    @test expectation == repeat([proba], n)
    @test TMLE.maybelogit(expectation, typeof(mach.model), target_scitype(mach.model)) == TMLE.logit(expectation)

    # Probabilistic Regressor
    y = rand(n)
    mach = machine(ConstantRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = predict(mach)
    expectation = TMLE.expected_value(ŷ, typeof(mach.model), target_scitype(mach.model))
    @test expectation ≈ repeat([mean(y)], n) atol=1e-10
    @test TMLE.maybelogit(expectation, typeof(mach.model), target_scitype(mach.model)) == expectation

    # Deterministic Regressor
    mach = machine(LinearRegressor(), X, y)
    fit!(mach; verbosity=0)
    ŷ = predict(mach)
    expectation = TMLE.expected_value(ŷ, typeof(mach.model), target_scitype(mach.model))
    @test expectation == ŷ
    @test TMLE.maybelogit(expectation, typeof(mach.model), target_scitype(mach.model)) == expectation

end

@testset "Test interaction_combinations" begin
    # With 1 treatment variable
    query = (t₁=["a", "b"],)
    combinations = TMLE.interaction_combinations(query)
    expected_comb = [(t₁ = "a",),
                     (t₁ = "b",)]
    for (i, comb) in enumerate(combinations)
        @test comb == expected_comb[i]
    end

    # With 3 treatment variables
    query = (t₁=["a", "b"], t₂ = ["c", "d"], t₃ = [true, false])
    combinations = TMLE.interaction_combinations(query)
    expected_comb = [(t₁ = "a", t₂ = "c", t₃ = true),
                     (t₁ = "b", t₂ = "c", t₃ = true),
                     (t₁ = "a", t₂ = "d", t₃ = true),
                     (t₁ = "b", t₂ = "d", t₃ = true),
                     (t₁ = "a", t₂ = "c", t₃ = false),
                     (t₁ = "b", t₂ = "c", t₃ = false),
                     (t₁ = "a", t₂ = "d", t₃ = false),
                     (t₁ = "b", t₂ = "d", t₃ = false)]
    for (i, comb) in enumerate(combinations)
        @test comb == expected_comb[i]
    end
end

@testset "Test indicator_fns" begin
    # For 1 treatment variable
    query = (t₁=["a", "b"],)
    indicators = TMLE.indicator_fns(query)
    @test indicators == Dict(
        (t₁ = "b",) => -1,
        (t₁ = "a",) => 1
    )

    # For 2 treatment variables:
    # The "case" treatment is: (true, false) 
    query = (t₁=[true, false], t₂ = [false, true])
    indicators = TMLE.indicator_fns(query)
    @test indicators == Dict(
        (t₁ = 1, t₂ = 0) => 1,
        (t₁ = 0, t₂ = 0) => -1,
        (t₁ = 1, t₂ = 1) => -1,
        (t₁ = 0, t₂ = 1) => 1
    )

    # For 3 treatment variables:
    # The "case" treatment is: (a, c, true) 
    query = (t₁=["a", "b"], t₂ = ["c", "d"], t₃ = [true, false])
    indicators = TMLE.indicator_fns(query)
    @test indicators == Dict(
        (t₁ = "b", t₂ = "c", t₃ = 1) => -1, # (-1)^{2}=1
        (t₁ = "a", t₂ = "c", t₃ = 1) => 1,  # (-1)^{0}=1
        (t₁ = "b", t₂ = "d", t₃ = 0) => -1, # (-1)^{3}=-1
        (t₁ = "b", t₂ = "c", t₃ = 0) => 1,  # (-1)^{2}=1
        (t₁ = "a", t₂ = "d", t₃ = 1) => -1, # (-1)^{1}=-1
        (t₁ = "a", t₂ = "c", t₃ = 0) => -1, # (-1)^{1}=-1
        (t₁ = "a", t₂ = "d", t₃ = 0) => 1,  # (-1)^{2}=1
        (t₁ = "b", t₂ = "d", t₃ = 1) => 1   # (-1)^{2}=1
    )
end

@testset "Test adapt" begin
    T = (a=1,)
    @test TMLE.adapt(T) == 1

    T = (a=1, b=2)
    @test TMLE.adapt(T) == T
end

@testset "Test compute_covariate" begin
    # First case: 1 categorical variable
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    T = (t₁ = categorical(["a", "b", "c", "a", "a", "b", "a"]),)
    W = MLJ.table(rand(7, 3))

    Gmach = machine(ConstantClassifier(), 
                    W, 
                    TMLE.adapt(T))
    fit!(Gmach, verbosity=0)

    indicators = TMLE.indicator_fns((t₁=["a", "b"],))

    cov = TMLE.compute_covariate(Gmach, W, T, indicators)
    @test cov == [1.75,
                 -3.5,
                 0.0,
                 1.75,
                 1.75,
                 -3.5,
                 1.75]

    # Second case: 2 binary variables
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    T = (t₁ = categorical([1, 0, 0, 1, 1, 1, 0]),
         t₂ = categorical([1, 1, 1, 1, 1, 0, 0]))
    W = MLJ.table(rand(7, 3))

    Gmach = machine(FullCategoricalJoint(ConstantClassifier()), 
                    W, 
                    T)
    fit!(Gmach, verbosity=0)

    indicators = TMLE.indicator_fns((t₁=[1, 0], t₂ = [1, 0]))

    cov = TMLE.compute_covariate(Gmach, W, T, indicators)
    @test cov == [2.3333333333333335,
                 -3.5,
                 -3.5,
                 2.3333333333333335,
                 2.3333333333333335,
                 -7.0,
                 7.0]

    # Third case: 3 mixed categorical variables
    # Using a trivial classifier
    # that outputs the proportions of of the classes
    T = (t₁ = categorical(["a", "a", "b", "b", "c", "b", "b"]),
         t₂ = categorical([3, 2, 1, 1, 2, 2, 2]),
         t₃ = categorical([true, false, true, false, false, false, false]))
    W = MLJ.table(rand(7, 3))

    Gmach = machine(FullCategoricalJoint(ConstantClassifier()), 
                    W, 
                    T)
    fit!(Gmach, verbosity=0)

    indicators = TMLE.indicator_fns((t₁=["a", "b"], t₂ = [1, 2], t₃ = [true, false]))

    cov = TMLE.compute_covariate(Gmach, W, T, indicators)
    @test cov == [0,
                  7.0,
                 -7,
                  7,
                  0,
                 -3.5,
                 -3.5]
end

@testset "Test compute_offset" begin
    n = 10
    X = rand(n, 3)

    # When Y is binary
    y = categorical([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    mach = machine(ConstantClassifier(), MLJ.table(X), y)
    fit!(mach, verbosity=0)
    # Should be equal to logit(Ê[Y|X])= logit(4/10) = -0.4054651081081643
    @test TMLE.compute_offset(mach, X) == repeat([-0.4054651081081643], n)

    # When Y is continuous
    y = [1., 2., 3, 4, 5, 6, 7, 8, 9, 10]
    mach = machine(MLJ.DeterministicConstantRegressor(), MLJ.table(X), y)
    fit!(mach, verbosity=0)
    # Should be equal to Ê[Y|X] = 5.5
    @test TMLE.compute_offset(mach, X) == repeat([5.5], n)
    
end

@testset "Test logit" begin
    @test TMLE.logit([0.4, 0.8, 0.2]) ≈ [
        -0.40546510810,
        1.38629436112,
        -1.38629436112
    ]
    @test TMLE.logit([1, 0]) == [Inf, -Inf]
end


@testset "Test compute_fluctuation" begin
    n = 10
    W = (w₁=ones(n),)
    T = (t₁=categorical([1, 1, 1, 0, 0, 0, 0, 1, 1, 0]),
         t₂=categorical([0, 0, 1, 0, 0, 1, 1, 1, 0, 0]))
    y = categorical([1, 1, 0, 0, 1, 0 , 1, 0, 0, 0])
    query = (t₁=[1, 0], t₂ = [1, 0])
    indicators = TMLE.indicator_fns(query)

    # Fit encoder
    Hmach = machine(OneHotEncoder(features=[:t₁, :t₂], drop_last=true), T)
    fit!(Hmach, verbosity=0)
    Thot = transform(Hmach)
    # Fit Q̅
    X = merge(Thot, W)
    Q̅mach = machine(ConstantClassifier(), X, y)
    fit!(Q̅mach, verbosity=0)
    # Fit G
    Gmach = machine(FullCategoricalJoint(ConstantClassifier()), W, T)
    fit!(Gmach, verbosity=0)
    # Fit Fluctuation
    offset = TMLE.compute_offset(Q̅mach, X)
    covariate = TMLE.compute_covariate(Gmach, W, T, indicators)
    Xfluct = (covariate=covariate, offset=offset)
    Fmach = machine(LinearBinaryClassifier(fit_intercept=false, offsetcol=:offset), Xfluct, y)
    fit!(Fmach, verbosity=0)

    # We are using constant classifiers
    # The offset is equal to: -0.40546510810 all the time
    # The covariate is ≈ [-3.3, -3.3, 5., 3.3, 3.3, -5., -5., 5., -3.3, 3.3]
    # The fluctuation value is equal to Ê[Y|W, T] where Ê is computed via Fmach
    # The coefficient for the fluctuation seems to be -0.222835 here 
    expected_mean(cov) = TMLE.expit(-0.40546510 - 0.22283549318*cov)
    # Let's look at the different counterfactual treatments
    # T₁₁: cov=5.
    T₁₁ = (t₁=categorical(ones(n), levels=levels(T[1])), t₂=categorical(ones(n), levels=levels(T[2])))
    fluct = TMLE.compute_fluctuation(Fmach, Q̅mach, Gmach, indicators, W, T₁₁, X)
    @test fluct ≈ repeat([expected_mean(5.)], n) atol=1e-5
    # T₁₀: cov=-3.333333
    T₁₀ = (t₁=categorical(ones(n), levels=[0, 1]), t₂=categorical(zeros(n), levels=[0, 1]))
    fluct = TMLE.compute_fluctuation(Fmach, Q̅mach, Gmach, indicators, W, T₁₀, X)
    @test fluct ≈ repeat([expected_mean(-3.333333)], n) atol=1e-5
    # T₀₁: cov=-5.
    T₀₁ = (t₁=categorical(zeros(n), levels=[0, 1]), t₂=categorical(ones(n), levels=[0, 1]))
    fluct = TMLE.compute_fluctuation(Fmach, Q̅mach, Gmach, indicators, W, T₀₁, X)
    @test fluct ≈ repeat([expected_mean(-5.)], n) atol=1e-5
    # T₀₀: cov=3.333333
    T₀₀ = (t₁=categorical(zeros(n), levels=[0, 1]), t₂=categorical(zeros(n), levels=[0, 1]))
    fluct = TMLE.compute_fluctuation(Fmach, Q̅mach, Gmach, indicators, W, T₀₀, X)
    @test fluct ≈ repeat([expected_mean(3.333333)], n) atol=1e-5

end

@testset "Test log_over_threshold" begin
    n = 10000
    rng = StableRNG(123)
    T = (t=categorical(rand(rng, Bernoulli(0.001), n)),)
    W = MLJ.table(rand(rng, n, 3))
    y = rand(rng, n)

    query = (t=[true, false],)

    Q̅ = LinearRegressor()
    G = LogisticClassifier()
    tmle = TMLEstimator(Q̅, G, query)

    mach = machine(tmle, T, W, y)
    fit!(mach, verbosity=0)
    @test length(report(mach).extreme_propensity_idx) == 12
end
end;

true