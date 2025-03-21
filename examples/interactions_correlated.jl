#=
# Interaction Estimation

In this example we aim to estimate the average interaction effect of two, potentially correlated, 
treatment variables `T1` and `T2` on an outcome `Y`.

## Data Generating Process

Let's consider the following data generating process (we first define some helper functions).
=#

using Distributions
using Random
using DataFrames
using Statistics
using CategoricalArrays
using TMLE
using CairoMakie
using MLJXGBoostInterface

Random.seed!(123)

μT(w) = [w, w]

μY(t, w) = 1 + 10t[1] - 3t[2] * t[1] * w

#=
`W`, the confounding variable, follows a uniform distribution.
=#

generate_W(n) = rand(Uniform(0, 1), n)


#=
`T1` and `T2` are generated via a copula method through a multivariate normal to induce some statistical dependence (via σ).
=#

function generate_T(W, n; σ=0.5, threshold=0)
    covariance = [
        1. σ
        σ 1.
    ]
    T = zeros(Bool, 2, n)
    for i in 1:n
        dTi = MultivariateNormal(μT(W[i]), covariance)
        T[:, i] = rand(dTi) .> threshold
    end
    return T
end

#=
Finally, `Y` is generated through a simple linear model with an interaction term.
=#

function generate_Y(T, W, n; σY=1)
    Y = zeros(Float64, n)
    for i in 1:n
        dY = Normal(μY(T[:, i], W[i]), σY)
        Y[i] = rand(dY)
    end
    return Y
end

#=
Importantly, the average interaction effect between `T1` and `T2` is thus ``-3 \cdot \mathbb{E}[W] = -1.5``

We will generate a full dataset with the following function.
=#

function generate_dataset(;n=1000, σ=0.5, threshold=0., σY=1)
    # Generate
    W = generate_W(n)
    T = generate_T(W, n; σ=σ, threshold=threshold)
    Y = generate_Y(T, W, n; σY=σY)
    # Make categorical treatments
    T = permutedims(T)
    T1 = categorical(T[:, 1])
    T2 = categorical(T[:, 2])
    # Make DataFrame
    return DataFrame(W=W, T1=T1, T2=T2, Y=Y)
end

dataset = generate_dataset()

#=
Let's verify that each treatment level is sufficiently present in the dataset (≈positivity).
=#

combine(groupby(dataset, [:T1, :T2]), proprow => :JOINT_TREATMENT_FREQ)

#=
And that `T1` and `T2` are indeed correlated.
=#

treatment_correlation(dataset) = cor(unwrap.(dataset.T1), unwrap.(dataset.T2))
@assert treatment_correlation(dataset) > 0.3 #hide
treatment_correlation(dataset)

#=
## Estimation

We can now proceed to estimation using TMLE and default (linear) models. 

Interactions are defined via the `AIE` function.
=#

Ψ = AIE(
    outcome=:Y,
    treatment_values= (
        T1=(case=1, control=0), 
        T2=(case=1, control=0)
    ),
    treatment_confounders = (
        T1=[:W],
        T2=[:W],
    )
)
estimator = TMLEE(weighted=true)
result, _ = estimator(Ψ, dataset; verbosity=0)
@assert pvalue(significance_test(result)) > 0.05 #hide
significance_test(result)

#=
The true effect size is thus covered by the confidence interval.

## Varying levels of correlation

We now vary the correlation level between T1 and T2 to observe how it affects the estimation results. 
First, let's see how the parameter σ affects the correlation between T1 and T2.
=#

function plot_correlations(;σs = 0.1:0.1:1, n=1000, threshold=0., σY=1.)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="σ", ylabel="Correlation(T1, T2)")
    correlations = map(σs) do σ
        dataset = generate_dataset(;n=n, σ=σ, threshold=threshold, σY=σY)
        return treatment_correlation(dataset)
    end
    scatter!(ax, σs, correlations, color=:blue)
    return fig
end

σs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999]
plot_correlations(;σs=σs, n=10_000)

#=
As expected, the correlation between T1 and T2 increases with σ. Let's see how this affects estimation, 
for this, we will vary both the dataset size and the correlation level.
=#

function estimate_across_correlation_levels(σs; n=1000, estimator=TMLEE(weighted=true))
    results = []
    for σ in σs
        dataset = generate_dataset(n=n, σ=σ)
        result, _ = estimator(Ψ, dataset; verbosity=0)
        push!(results, result)
    end
    Ψ̂s = TMLE.estimate.(results)
    errors = last.(confint.(significance_test.(results))) .- Ψ̂s
    return Ψ̂s, errors
end

function estimate_across_sample_sizes_and_correlation_levels(ns, σs; estimator=TMLEE(weighted=true))
    results = []
    for n in ns
        Ψ̂s, errors = estimate_across_correlation_levels(σs; n=n, estimator=estimator)
        push!(results, (Ψ̂s, errors))
    end
    return results
end

function plot_across_sample_sizes_and_correlation_levels(results, ns, σs; title="Estimation via TMLE (GLMs)")
    fig = Figure(size=(800, 800))
    for (index, n) in enumerate(ns)
        Ψ̂s, errors = results[index]
        ax = if n == last(ns)
            Axis(fig[index, 1], xlabel="σ", ylabel="AIE\n(n=$n)")
        else
            Axis(fig[index, 1], ylabel="AIE\n(n=$n)", xticklabelsvisible=false)
        end
        errorbars!(ax, σs, Ψ̂s, errors, color = :blue, whiskerwidth = 10)
        scatter!(ax, σs, Ψ̂s, color=:red, markersize=10)
        hlines!(ax, [-1.5], color=:green, linestyle=:dash)
    end
    Label(fig[0, :], title; tellwidth=false, fontsize=24)
    return fig
end

ns = [100, 1000, 10_000, 100_000, 1_000_000]
σs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999]
results = estimate_across_sample_sizes_and_correlation_levels(ns, σs)
plot_across_sample_sizes_and_correlation_levels(results, ns, σs; title="Estimation via TMLE (GLMs)")

#=
First, notice that only extreme correlations (>0.9) tend to blow up the size of the confidence intervals. This implies that statistical power may be limited in such circumstances.

Furthermore, and perhaps unexpectedly, coverage decreases as sample size grows for larger correlations. Since we have used simple linear models until now, 
this could be due to model misspecification. We can verify this by using a more flexible modelling strategy. Here we will use XGBoost 
(with tree_method="hist" to speed things up a little).
=#

xgboost_estimator = TMLEE(
    models=default_models(G=XGBoostClassifier(tree_method="hist"), Q_continuous=XGBoostRegressor(tree_method="hist")),
    weighted=true
)
xgboost_results = estimate_across_sample_sizes_and_correlation_levels(ns, σs, estimator=xgboost_estimator)
plot_across_sample_sizes_and_correlation_levels(xgboost_results, ns, σs; title="Estimation via TMLE (XGboost)")

#=
As expected, XGBoost improves estimation performance in the asymptotic regime, furthermore, 
the correlation between T1 and T2 seems harmless (except when σ > 0.9 as before). 

However, the performance is degraded for smaller sample sizes, likely due to overfitting. A more agressive strategy 
relying on Super-Learning or cross-validation would probably be beneficial in this case.
=#
