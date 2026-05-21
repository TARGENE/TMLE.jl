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
    Ycol     = pop[!, outcome_col]
    cases    = findall(Ycol .== 1)
    controls = findall(Ycol .== 0)
    if length(cases) < n_case
        throw(ArgumentError("Not enough cases: have $(length(cases)), need $n_case"))
    end
    if length(controls) < n_ctl
        throw(ArgumentError("Not enough controls: have $(length(controls)), need $n_ctl"))
    end
    ix_case = shuffle(rng, cases)[1:n_case]
    ix_ctl  = shuffle(rng, controls)[1:n_ctl]
    ix = vcat(ix_case, ix_ctl)
    ix = shuffle(rng, ix)
    sub_pop = pop[ix, :]
    sub_pop.A = categorical(Bool.(sub_pop.A))
    sub_pop.Y = categorical(Bool.(sub_pop.Y))
    return sub_pop
end

function pY_given_A_W(A, W; α=-3, β=log(2), γ=log(1.5))
    ηY = α .+ β .* A .+ γ .* W
    return 1 ./ (1 .+ exp.(-ηY))
end

@testset "CCW-TMLE / CCW-OSE bootstrapping test" begin
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
    # Canonical approach (no prevalence correction)
    tmle_std = Tmle(weighted=false)
    ose_std = Ose()
    # CCW-approach (with true prevalence)
    tmle_ccw = Tmle(prevalence=q₀, weighted=false)
    ose_ccw = Ose(prevalence=q₀)

    # Draw a series of biased samples of size n_sample
    n_sample = 10_000
    cc_prev = 0.2
    B = 30
    ccw_tmle_results = Vector{Any}(undef, B)
    std_tmle_results = Vector{Any}(undef, B)
    ccw_ose_results  = Vector{Any}(undef, B)
    std_ose_results  = Vector{Any}(undef, B)
    ccw_tmle_coverage = Vector{Bool}(undef, B)
    std_tmle_coverage = Vector{Bool}(undef, B)
    ccw_ose_coverage  = Vector{Bool}(undef, B)
    std_ose_coverage  = Vector{Bool}(undef, B)

    for i in 1:B
        sample = subsample_case_control(pop, n_sample, cc_prev, rng=Random.MersenneTwister(i))

        # TMLE: standard vs CCW
        std_tmle_result, _ = tmle_std(Ψ, sample; verbosity=0)
        ccw_tmle_result, _ = tmle_ccw(Ψ, sample; verbosity=0)
        std_tmle_results[i] = std_tmle_result.estimate
        ccw_tmle_results[i] = ccw_tmle_result.estimate
        # CCW-TMLE should be much less biased than standard TMLE
        @test abs(ccw_tmle_result.estimate - true_rd) <
              abs(std_tmle_result.estimate - true_rd) / 2
        lb, ub = confint(significance_test(ccw_tmle_result))
        ccw_tmle_coverage[i] = lb < true_rd < ub
        lb, ub = confint(significance_test(std_tmle_result))
        std_tmle_coverage[i] = lb < true_rd < ub

        # OSE: standard vs CCW
        std_ose_result, _ = ose_std(Ψ, sample; verbosity=0)
        ccw_ose_result, _ = ose_ccw(Ψ, sample; verbosity=0)
        std_ose_results[i] = std_ose_result.estimate
        ccw_ose_results[i] = ccw_ose_result.estimate
        # CCW-OSE should be much less biased than standard OSE
        @test abs(ccw_ose_result.estimate - true_rd) <
              abs(std_ose_result.estimate - true_rd) / 2
        lb, ub = confint(significance_test(ccw_ose_result))
        ccw_ose_coverage[i] = lb < true_rd < ub
        lb, ub = confint(significance_test(std_ose_result))
        std_ose_coverage[i] = lb < true_rd < ub
    end

    # On average, CCW versions outperform their standard counterparts
    @test (mean(ccw_tmle_results) - true_rd) < (mean(std_tmle_results) - true_rd)
    @test (mean(ccw_ose_results)  - true_rd) < (mean(std_ose_results)  - true_rd)
    # Coverage is improved as well
    @test mean(ccw_tmle_coverage) > mean(std_tmle_coverage)
    @test mean(ccw_tmle_coverage) > 0.80
    @test mean(ccw_ose_coverage)  > mean(std_ose_coverage)
    @test mean(ccw_ose_coverage)  > 0.80
end

end
true