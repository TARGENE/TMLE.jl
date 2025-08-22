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
    # Ensure A,Y are Int 0/1 then categorical (levels 0,1) for MLJ binary classifier:
    if !(eltype(sub_pop.A) <: Integer)
        sub_pop.A = Int.(sub_pop.A)
    end
    if !(eltype(sub_pop.Y) <: Integer)
        sub_pop.Y = Int.(sub_pop.Y)
    end
    sub_pop.A = categorical(sub_pop.A; compress=true)
    sub_pop.Y = categorical(sub_pop.Y; compress=true)
    return sub_pop
end

function pY_given_A_W(A, W; α=-3, β=log(2), γ=log(1.5))
    ηY = α .+ β .* A .+ γ .* W
    return 1 ./ (1 .+ exp.(-ηY))
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
    ηY = α .+ β .* A .+ γ .* W
    pY = 1 ./ (1 .+ exp.(-ηY))
    Y = rand.(Bernoulli.(pY))
    pop = DataFrame(W=W, A=A, Y=Y)
    q₀ = mean(pop.Y .== 1)

    # Obtain the true risk difference (ATE)
    Y_1 = pY_given_A_W(1, pop.W)
    Y_0 = pY_given_A_W(0, pop.W)
    true_rd = mean(Y_1 .- Y_0)

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
    n_sample = 100_000
    cc_prev = 0.2
    ccw_tmle_results = Vector{Any}()
    std_tmle_results = Vector{Any}()
    ccw_coverage = Vector{Bool}()

    for i in 1:30
        sample = subsample_case_control(pop, n_sample, cc_prev, rng=Random.MersenneTwister(i))
        std_result, _ = tmle_std(Ψ, sample; verbosity=0)
        ccw_result, _ = tmle_ccw(Ψ, sample; verbosity=0)
        push!(std_tmle_results, std_result.estimate)
        push!(ccw_tmle_results, ccw_result.estimate)
        # Compare bias
        std_bias = abs(std_result.estimate - true_rd)
        ccw_bias = abs(ccw_result.estimate - true_rd)
        # CCW-TMLE should be much less biased than standard TMLE
        @test ccw_bias < std_bias / 2
        # CCW-TMLE confidence interval should cover the truth
        lb, ub = confint(significance_test(ccw_result))
        push!(ccw_coverage, lb < true_rd < ub)
    end
    # See if, on average, CCW-TMLE outperforms standard TMLE
    @test (mean(ccw_tmle_results) - true_rd) < (mean(std_tmle_results) - true_rd)
end

end
true