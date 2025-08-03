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
    sub_pop = pop[vcat(ix_case, ix_ctl), :]
    sub_pop.A = categorical(sub_pop.A)
    sub_pop.Y = categorical(sub_pop.Y)
    return sub_pop
end

@testset "CCW-TMLE reduces bias compared to standard TMLE under case-control sampling" begin
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

    # Estimate true causal effect via g-computation
    pop2 = copy(pop)
    pop2.A = Float64.(pop2.A)
    pop2.W = Float64.(pop2.W)
    pop2.Y = categorical(pop2.Y)
    Xpop = DataFrame(A=pop2.A, W=pop2.W)
    ypop = pop2.Y
    logreg = LogisticClassifier()
    mach   = machine(logreg, Xpop, ypop)
    fit!(mach)
    df1 = DataFrame(A = ones(Float64, Npop), W = pop2.W)
    df0 = DataFrame(A = zeros(Float64, Npop), W = pop2.W)
    π1 = pdf.(predict(mach, df1), 1)
    π0 = pdf.(predict(mach, df0), 1)
    true_rd = mean(π1 .- π0)

    # Draw a severely biased case-control sample (e.g., 20% cases, 80% controls)
    n_sample = 100_000
    cc_prev = 0.2
    sample = subsample_case_control(pop, n_sample, cc_prev)

    # Define ATE estimand
    Ψ = ATE(
        outcome=:Y,
        treatment_values=(A=(case=true, control=false),),
        treatment_confounders=(A=[:W],)
    )

    # Standard TMLE (no prevalence correction)
    tmle_std = Tmle(weighted=false)
    std_result, _ = tmle_std(Ψ, sample; verbosity=0)

    # CCW-TMLE (with true prevalence)
    tmle_ccw = Tmle(prevalence=q₀, weighted=false)
    ccw_result, _ = tmle_ccw(Ψ, sample; verbosity=0)

    # Compare bias
    std_bias = abs(std_result.estimate - true_rd)
    ccw_bias = abs(ccw_result.estimate - true_rd)

    @info "True causal RD" true_rd
    @info "Standard TMLE estimate" std_result.estimate
    @info "CCW-TMLE estimate" ccw_result.estimate
    @info "Standard TMLE bias" std_bias
    @info "CCW-TMLE bias" ccw_bias

    # CCW-TMLE should be much less biased than standard TMLE
    @test ccw_bias < std_bias / 2

    # Both estimates should be finite
    @test isfinite(std_result.estimate)
    @test isfinite(ccw_result.estimate)

    # CCW-TMLE confidence interval should cover the truth
    lb, ub = confint(significance_test(ccw_result))
    @test lb < true_rd < ub
end

end