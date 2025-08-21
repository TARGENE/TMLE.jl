#=
# Case–Control Sampling and Prevalence-Weighted TMLE

This example illustrates:
1. How *case–control* sampling induces sampling bias for population
   target parameters.
2. How providing the *true population prevalence* to TMLE (case–control weighted TMLE)
   corrects that bias.
3. Sensitivity to *misspecification* of the assumed population prevalence.

We generate a very large "source population",
then repeatedly draw case–control samples at various *sampling* prevalences `q`. 
We compare:
- Canonical TMLE (ignores population prevalence)
- CCW–TMLE (with known true population prevalence `q₀`)

The target estimand is the Average Treatment Effect (risk difference) for a binary
treatment `A` on binary outcome `Y` with a binary confounder `W`.
=#

using Random, DataFrames, Distributions, Statistics
using MLJBase
using CategoricalArrays
using CairoMakie
using TMLE

#=
# ---------------------------------------------------------
# Data Generating Process
# ---------------------------------------------------------
# W ~ Bernoulli(0.5)
# logit P(A=1|W) = −0.2 + 0.8 W
# logit P(Y=1|A,W) = α + β A + γ W
# Choose α to give marginal prevalence around 5%
=#

α, β, γ = -3.0, log(2.0), log(1.5)
pA_given_W(w) = 1 / (1 + exp(-(-0.2 + 0.8*w)))
pY_given_A_W(a, w) = 1 / (1 + exp(-(α + β*a + γ*w)))

function generate_population(N::Int)
    W = rand(Bernoulli(0.5), N)
    A = [rand(Bernoulli(pA_given_W(w))) for w in W]
    Y = [rand(Bernoulli(pY_given_A_W(a, w))) for (a, w) in zip(A, W)]
    DataFrame(W=W, A=A, Y=Y)
end

function true_risk_difference(pop::DataFrame)
    W = pop.W
    π1 = pY_given_A_W.(1, W)
    π0 = pY_given_A_W.(0, W)
    mean(π1 .- π0)
end

function subsample_case_control(
    pop::DataFrame,
    n::Int,
    sampling_prev::Float64;
    outcome_col::Symbol = :Y,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    n_case = round(Int, sampling_prev * n)
    n_ctl  = n - n_case
    y = pop[!, outcome_col]
    case_idx = findall(y .== 1)
    ctl_idx  = findall(y .== 0)
    length(case_idx) < n_case && error("Not enough cases.")
    length(ctl_idx)  < n_ctl  && error("Not enough controls.")
    sel_cases = shuffle(rng, case_idx)[1:n_case]
    sel_ctls  = shuffle(rng, ctl_idx)[1:n_ctl]
    sub = pop[vcat(sel_cases, sel_ctls), :]
    sub.A = categorical(sub.A)
    sub.Y = categorical(sub.Y)
    sub
end


ate_spec() = ATE(
    outcome = :Y,
    treatment_values = (A=(case=1, control=0),),
    treatment_confounders = (A=[:W],)
)

function estimate_standard_and_ccw(sample::DataFrame, q0::Float64)
    Ψ = ate_spec()
    tmle_plain = Tmle(weighted=false)
    tmle_ccw   = Tmle(prevalence=q0, weighted=false)
    est_plain, _ = tmle_plain(Ψ, sample; verbosity=0)
    est_ccw, _   = tmle_ccw(Ψ, sample; verbosity=0)
    return est_plain, est_ccw
end

function ribbon!(ax, x, y, ylow, yhigh; color, label=nothing)
    band!(ax, x, ylow, yhigh, color=(color, 0.25))
    lines!(ax, x, y, color=color, label=label, linewidth=2)
end

function sampling_bias_analysis(
    pop::DataFrame,
    n::Int,
    sampling_range::Tuple{Float64,Float64},
    n_studies::Int,
    q0::Float64,
    true_Ψ::Float64;
    rng=Random.GLOBAL_RNG
)
    q_min, q_max = sampling_range
    qs = exp.(range(log(q_min), log(q_max), length=n_studies))

    results = DataFrame(q=Float64[],
                        method=String[],
                        estimate=Float64[],
                        lower=Float64[],
                        upper=Float64[])

    for q in qs
        sample = subsample_case_control(pop, n, q; rng=rng)
        est_plain, est_ccw = estimate_standard_and_ccw(sample, q0)
        ci_plain = confint(significance_test(est_plain))
        ci_ccw   = confint(significance_test(est_ccw))
        push!(results, (q, "TMLE", est_plain.estimate, ci_plain[1], ci_plain[2]))
        push!(results, (q, "CCW–TMLE", est_ccw.estimate, ci_ccw[1], ci_ccw[2]))
    end

    fig = Figure(resolution=(800,500))
    ax = Axis(fig[1,1], title="Sampling Bias: Standard vs CCW–TMLE",
              xlabel="Sampling prevalence q",
              ylabel="ATE (risk difference)")
    df_plain = filter(:method => ==("TMLE"), results)
    ribbon!(ax, df_plain.q, df_plain.estimate, df_plain.lower, df_plain.upper;
            color=:dodgerblue, label="Standard TMLE")
    df_ccw = filter(:method => ==("CCW–TMLE"), results)
    ribbon!(ax, df_ccw.q, df_ccw.estimate, df_ccw.lower, df_ccw.upper;
            color=:orange, label="CCW–TMLE")
    hlines!(ax, [true_Ψ], color=:black, linestyle=:dash, label="True ATE")
    vlines!(ax, [q0], color=:red, linestyle=:dot, label="True q₀")
    axislegend(ax, position=:rb)
    return results, fig
end

function sampling_bias_misspecification(
    pop::DataFrame,
    n::Int,
    sampling_range::Tuple{Float64,Float64},
    n_studies::Int,
    true_q0::Float64,
    assumed_q0s::Vector{Float64},
    true_Ψ::Float64;
    rng=Random.GLOBAL_RNG
)
    qs = exp.(range(log(sampling_range[1]), log(sampling_range[2]), length=n_studies))
    results = DataFrame(q=Float64[],
                        assumed_q0=Float64[],
                        method=String[],
                        estimate=Float64[],
                        lower=Float64[],
                        upper=Float64[])

    Ψ = ate_spec()

    for q in qs
        sample = subsample_case_control(pop, n, q; rng=rng)

        est_plain, _ = Tmle(weighted=false)(Ψ, sample; verbosity=0)
        ci_plain = confint(significance_test(est_plain))
        push!(results, (q, NaN, "TMLE", est_plain.estimate, ci_plain[1], ci_plain[2]))

        for q_assumed in assumed_q0s
            est_ccw, _ = Tmle(prevalence=q_assumed, weighted=false)(Ψ, sample; verbosity=0)
            ci_ccw = confint(significance_test(est_ccw))
            push!(results, (q, q_assumed, "CCW–TMLE", est_ccw.estimate, ci_ccw[1], ci_ccw[2]))
        end
    end

    fig = Figure(resolution=(850,500))
    ax = Axis(fig[1,1], title="Prevalence Misspecification Sensitivity",
              xlabel="Sampling prevalence q",
              ylabel="ATE (risk difference)")

    df_plain = filter(:method => ==("TMLE"), results)
    ribbon!(ax, df_plain.q, df_plain.estimate, df_plain.lower, df_plain.upper;
            color=:gray40, label="TMLE")

    colors = (:green, :magenta, :brown, :teal, :purple)
    for (k, q_assumed) in enumerate(assumed_q0s)
        df_ccw = filter(row -> row.assumed_q0 == q_assumed, results)
        ribbon!(ax, df_ccw.q, df_ccw.estimate, df_ccw.lower, df_ccw.upper;
                color=colors[mod1(k, length(colors))],
                label="CCW q₀=$(round(q_assumed, sigdigits=3))")
    end

    hlines!(ax, [true_Ψ], color=:black, linestyle=:dash, label="True ATE")
    vlines!(ax, [true_q0], color=:red, linestyle=:dot, label="True q₀")
    axislegend(ax, position=:rb)
    return results, fig
end

Random.seed!(42)
N_pop = 2_000_000
pop = generate_population(N_pop)
q₀ = mean(pop.Y)
true_RD = true_risk_difference(pop)
@info "True population prevalence q₀ = $(round(q₀, sigdigits=4))"
@info "True causal risk difference ≈ $(round(true_RD, sigdigits=5))"

results_basic, fig_basic = sampling_bias_analysis(pop, 100_000, (0.01, 0.5), 8, q₀, true_RD)
results_miss, fig_miss = sampling_bias_misspecification(pop, 60_000, (0.01, 0.4), 6, q₀,
                                                        [q₀, q₀*0.5, q₀*2], true_RD)

display(fig_basic)
display(fig_miss)

#=
Interpretation:
- Standard TMLE estimates drift as sampling prevalence deviates from q₀ because the
  empirical distribution over- or under-represents cases.
- CCW–TMLE with the true q₀ remains stable across sampling prevalences.
- Misspecifying q₀ introduces bias whose direction depends on whether the assumed q₀
  over- or underestimates the true prevalence.

This demonstrates how supplying external information (the true population prevalence) can
recover the true population structure in biased (case–control) sampling designs.
=#