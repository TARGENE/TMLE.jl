module TestCCWTMLEBiasReduction

using Test
using TMLE
using DataFrames
using CategoricalArrays
using Random
using Distributions
using MLJBase
using MLJLinearModels
using Statistics
using CSV

# Helper: Draw a case-control sample with specified prevalence
function subsample_case_control(
    pop::DataFrame,
    n::Int,
    prevalence::Float64;
    outcome_col::Symbol = :Y,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    n_case = round(Int, prevalence * n)
    n_ctl  = n - n_case

    ycol = pop[!, outcome_col]
    cases = findall(ycol .== 1)
    controls = findall(ycol .== 0)
    if length(cases) < n_case
        throw(ArgumentError("Not enough cases for $outcome_col: have $(length(cases)), need $n_case"))
    end
    if length(controls) < n_ctl
        throw(ArgumentError("Not enough controls for $outcome_col: have $(length(controls)), need $n_ctl"))
    end
    ix_case = shuffle(rng, cases)[1:n_case]
    ix_ctl  = shuffle(rng, controls)[1:n_ctl]
    ix = shuffle(rng, vcat(ix_case, ix_ctl))

    sub_pop = pop[ix, :]
    sub_pop.A = categorical(Bool.(sub_pop.A))
    sub_pop[!, outcome_col] = categorical(Bool.(sub_pop[!, outcome_col]))

    return sub_pop
end

function pY_given_A_W(A, W; α=-3, β=log(2), γ=log(1.5))
    ηY = α .+ β .* A .+ γ .* W
    return 1 ./ (1 .+ exp.(-ηY))
end

function make_population(Npop::Int)
    W = rand(Bernoulli(0.5), Npop)
    ηA = -0.2 .+ 0.8 .* W
    pA = 1 ./ (1 .+ exp.(-ηA))
    A = rand.(Bernoulli.(pA))

    pY1 = pY_given_A_W(A, W; α=-3.0, β=log(2.0), γ=log(1.5))
    pY2 = pY_given_A_W(A, W; α=-2.2, β=log(1.4), γ=log(1.8))

    Y1 = rand.(Bernoulli.(pY1))
    Y2 = rand.(Bernoulli.(pY2))

    return DataFrame(W=W, A=A, Y1=Y1, Y2=Y2)
end

@testset "CCW-TMLE bootstrapping test" begin
    Random.seed!(42)
    Npop = 2_000_000
    # Simulate population
    W = rand(Bernoulli(0.5), Npop)
    ηA = -0.2 .+ 0.8 .* W
    pA = 1 ./ (1 .+ exp.(-ηA))
    A = rand.(Bernoulli.(pA))
    α, β, γ = -3, log(2), log(1.5)
    pY = pY_given_A_W(A, W; α=α, β=β, γ=γ)
    Y = rand.(Bernoulli.(pY))
    pop = DataFrame(W=W, A=A, Y=Y)
    q₀ = mean(pop.Y .== 1)

    # Obtain the true risk difference (ATE)
    true_rd = mean(pY_given_A_W(1, pop.W) .- pY_given_A_W(0, pop.W))

    # Define ATE estimand
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(A=(case=true, control=false),),
        treatment_confounders=(A=[:W],)
    )
    # Standard TMLE (no prevalence correction)
    tmle_std = Tmle(weighted=false)
    # CCW-TMLE (with true prevalence)
    tmle_ccw = Tmle(prevalence=q₀, weighted=false)

    # Draw a series of biased samples of size n_sample
    n_sample = 10_000
    cc_prev = 0.2
    B = 30
    ccw_tmle_results = Vector{Any}(undef, B)
    std_tmle_results = Vector{Any}(undef, B)
    ccw_coverage = Vector{Bool}(undef, B)
    std_coverage = Vector{Bool}(undef, B)

    for i in 1:B
        sample = subsample_case_control(pop, n_sample, cc_prev, rng=Random.MersenneTwister(i))
        std_result, _ = tmle_std(Ψ, sample; verbosity=0)
        ccw_result, _ = tmle_ccw(Ψ, sample; verbosity=0)
        std_tmle_results[i] = std_result.estimate
        ccw_tmle_results[i] = ccw_result.estimate
        # Compare bias: CCW-TMLE should be much less biased than standard TMLE
        std_bias = abs(std_result.estimate - true_rd)
        ccw_bias = abs(ccw_result.estimate - true_rd)
        @test ccw_bias < std_bias / 2
        # Retrieve coverage
        lb, ub = confint(significance_test(ccw_result))
        ccw_coverage[i] = lb < true_rd < ub
        lb, ub = confint(significance_test(std_result))
        std_coverage[i] = lb < true_rd < ub
    end
    # See if, on average, CCW-TMLE outperforms standard TMLE
    @test (mean(ccw_tmle_results) - true_rd) < (mean(std_tmle_results) - true_rd)
    # Test coverage is improved as well
    @test mean(ccw_coverage) > mean(std_coverage)
    @test mean(ccw_coverage) > 0.80
end

@testset "Test multi-trait CCW run with prevalence dictionary" begin
    Random.seed!(42)
    pop = make_population(200_000)

    # For running full model, copy pop
    pop_copy = deepcopy(pop)
    pop_copy.A  = categorical(pop_copy.A)
    pop_copy.Y1 = categorical(pop_copy.Y1)
    pop_copy.Y2 = categorical(pop_copy.Y2)

    # True prevalences computed from the population
    prevalence_by_trait = Dict(
        :Y1 => mean(pop.Y1),
        :Y2 => mean(pop.Y2),
    )

    # Ground truth for each trait, using the parameters that generated them
    trait_params = Dict(
        :Y1 => (α = -3.0, β = log(2.0), γ = log(1.5)),
        :Y2 => (α = -2.2, β = log(1.4), γ = log(1.8)),
    )

    true_rd_by_trait = Dict{Symbol, Float64}()
    for trait in [:Y1, :Y2]
        p = trait_params[trait]
        true_rd_by_trait[trait] = mean(
            pY_given_A_W(1, pop.W; α=p.α, β=p.β, γ=p.γ) .-
            pY_given_A_W(0, pop.W; α=p.α, β=p.β, γ=p.γ)
        )
    end

    traits = [:Y1, :Y2]
    n_sample = 10_000
    B = 10

    for trait in traits
        trait_prev = prevalence_by_trait[trait]
        true_rd_trait = true_rd_by_trait[trait]

        Ψ = ATE(
            outcome = trait,
            treatment_values = (A = (case = true, control = false),),
            treatment_confounders = (A = [:W],)
        )

        tmle_std = Tmle(weighted=false)
        tmle_ccw = Tmle(prevalence=trait_prev, weighted=false)
        tmle_ccw_prev_dict = Tmle(prevalence=prevalence_by_trait, weighted=false)

        # Check on full population: dict-based prevalence vs scalar prevalence
        ccw_full_result, _ = tmle_ccw(Ψ, pop_copy; verbosity=0)
        prev_dict_full_result, _ = tmle_ccw_prev_dict(Ψ, pop_copy; verbosity=0)
        @test isapprox(ccw_full_result.estimate, prev_dict_full_result.estimate; atol=1e-3)

        std_estimates = Float64[]
        ccw_estimates = Float64[]
        prev_dict_estimates = Float64[]

        for b in 1:B
            sample = subsample_case_control(
                pop,
                n_sample,
                trait_prev;
                outcome_col = trait,
                rng = Random.MersenneTwister(1000 + b),
            )

            std_result, _ = tmle_std(Ψ, sample; verbosity=0)
            ccw_result, _ = tmle_ccw(Ψ, sample; verbosity=0)

            # Dictionary-based prevalence run
            prev_dict_result, _ = tmle_ccw_prev_dict(Ψ, sample; verbosity=0)

            push!(std_estimates, std_result.estimate)
            push!(ccw_estimates, ccw_result.estimate)
            push!(prev_dict_estimates, prev_dict_result.estimate)

            @test isfinite(std_result.estimate)
            @test isfinite(ccw_result.estimate)
            @test isfinite(prev_dict_result.estimate)
        end

        # Dict-based prevalence and scalar prevalence should agree closely
        @test isapprox(mean(prev_dict_estimates), mean(ccw_estimates); atol=1e-3)
        # Bias should be reduced with correct prevalence specified
        @test abs(mean(ccw_estimates) - true_rd_trait) < abs(mean(std_estimates) - true_rd_trait)
    end
end

end
true