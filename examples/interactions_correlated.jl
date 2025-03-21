#=
# Interaction Estimation

In this example we aim to estimate the average interaction effect of two, potentially correlated, 
treatment variables `T1` and `T2` on an outcome `Y`.

## Data Generating Process

Let's consider the following structural causal model where the shaded nodes represent the observed variables.

![interaction-graph](assets/interaction_graph.png)

In other words, only one confounding variable is observed (`W1`). This would be a major problem if we wanted to estimate the 
average treatment effect of `T1` or `T2` on `Y` separately. However, here, we are interested in interactions and thus `W1` is 
a sufficient adjustment set. This artificial situation is ubiquitous in genetics, where two main sources of confounding exist. 
Ancestry, can be estimated (here `W1`) and linkage disequilibrium is usually more challenging to address (here `W2`).

Let us first define some helper functions and import some libraries.
=#
using Distributions
using Random
using DataFrames
using Statistics
using CategoricalArrays
using TMLE
using CairoMakie
using MLJXGBoostInterface
using MLJBase
using MLJLinearModels
Random.seed!(123)

μT(w) = [sum(w), sum(w)]

μY(t, w) = 1 + 10t[1] - 3t[2] * t[1] * w

#=
We assume that `W1` and `W2`, the confounding variables, follow a uniform distribution.
=#

generate_W(n) = rand(Uniform(0, 1), 2, n)

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
        dTi = MultivariateNormal(μT(W[:, i]), covariance)
        T[:, i] = rand(dTi) .> threshold
    end
    return T
end

#=
Finally, `Y` is generated through a simple linear model with an interaction term.
=#

function generate_Y(T, W1, n; σY=1)
    Y = zeros(Float64, n)
    for i in 1:n
        dY = Normal(μY(T[:, i], W1[i]), σY)
        Y[i] = rand(dY)
    end
    return Y
end

#=
Importantly, the average interaction effect between `T1` and `T2` is thus ``-3 \mathbb{E}[W] = -1.5``.

We will generate a full dataset with the following function.
=#

function generate_dataset(;n=1000, σ=0.5, threshold=0., σY=1)

    W = generate_W(n)
    T = generate_T(W, n; σ=σ, threshold=threshold)

    W = permutedims(W)
    W1 = W[:, 1]
    W2 = W[:, 2]

    Y = generate_Y(T, W1, n; σY=σY)

    T = permutedims(T)
    T1 = categorical(T[:, 1])
    T2 = categorical(T[:, 2])

    return DataFrame(W1=W1, W2=W2, T1=T1, T2=T2, Y=Y)
end

dataset = generate_dataset()

first(dataset, 5)
#=
Let's verify that each treatment level is sufficiently present in the dataset (≈positivity).
=#

combine(groupby(dataset, [:T1, :T2]), proprow => :JOINT_TREATMENT_FREQ)

#=
And that `T1` and `T2` are indeed correlated.
=#

treatment_correlation(dataset) = cor(unwrap.(dataset.T1), unwrap.(dataset.T2))
@assert treatment_correlation(dataset) > 0.2 #hide
treatment_correlation(dataset)

#=
## Estimation

We can now proceed to estimation using TMLE and default (linear) models. 

Interactions are defined via the `AIE` function (note that we only set `W1` as a confounder).
=#

Ψ = AIE(
    outcome=:Y,
    treatment_values= (
        T1=(case=1, control=0), 
        T2=(case=1, control=0)
    ),
    treatment_confounders = [:W1]
)
linear_models = default_models(G=LogisticClassifier(lambda=0), Q_continuous=LinearRegressor())
estimator = TMLEE(models=linear_models, weighted=true)
result, _ = estimator(Ψ, dataset; verbosity=0)
@assert pvalue(significance_test(result, -1.5)) > 0.05 #hide
significance_test(result)

#=
The true effect size is thus covered by our confidence interval.

## Varying levels of correlation

We now vary the correlation level between `T1` and `T2` to observe how it affects the estimation results. 
First, let's see how the parameter σ affects the correlation between `T1` and `T2`.
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

σs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
plot_correlations(;σs=σs, n=10_000)

#=
As expected, the correlation between `T1` and `T2` increases with σ. Let's see how this affects estimation, 
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

function estimate_across_sample_sizes_and_correlation_levels(ns, σs; estimator=TMLEE(models=linear_models, weighted=true))
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
results = estimate_across_sample_sizes_and_correlation_levels(ns, σs; estimator=TMLEE(models=linear_models, weighted=true))
plot_across_sample_sizes_and_correlation_levels(results, ns, σs; title="Estimation via TMLE (GLMs)")

#=
First, notice that only extreme correlations (>0.9) tend to blow up the size of the confidence intervals. This implies that statistical power may be limited in such circumstances.

Furthermore, and perhaps unexpectedly, coverage decreases as sample size grows for larger correlations. Since we have used simple linear models until now, 
this could be due to model misspecification. We can verify this by using a more flexible modelling strategy. Here we will use XGBoost 
(with tree_method=`hist` to speed things up a little). Because this model is prone to overfitting we will also use cross-validation (this will take a few minutes).
=#

xgboost_estimator = TMLEE(
    models=default_models(G=XGBoostClassifier(tree_method="hist"), Q_continuous=XGBoostRegressor(tree_method="hist")),
    weighted=true,
    resampling=StratifiedCV(nfolds=3)
)
xgboost_results = estimate_across_sample_sizes_and_correlation_levels(ns, σs, estimator=xgboost_estimator)
plot_across_sample_sizes_and_correlation_levels(xgboost_results, ns, σs; title="Estimation via TMLE (XGboost)")

#=
As expected, XGBoost improves estimation performance in the asymptotic regime, furthermore, 
the correlation between `T1` and `T2` seems harmless (except when σ > 0.9 as before).
=#
