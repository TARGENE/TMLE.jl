```@meta
EditURL = "../../../examples/interactions_correlated.jl"
```

# Interaction Estimation

In this example we aim to estimate the average interaction effect of two, potentially correlated,
treatment variables `T1` and `T2` on an outcome `Y`.

## Data Generating Process

Let's consider the following structural causal model where the shaded nodes represent the observed variables.

![interaction-graph](../assets/interaction_graph.png)

In other words, only one confounding variable is observed (`W1`). This would be a major problem if we wanted to estimate the
average treatment effect of `T1` or `T2` on `Y` separately. However, here, we are interested in interactions and thus `W1` is
a sufficient adjustment set. This artificial situation is ubiquitous in genetics, where two main sources of confounding exist.
Ancestry, can be estimated (here `W1`) and linkage disequilibrium is usually more challenging to address (here `W2`).

Let us first define some helper functions and import some libraries.

````@example interactions_correlated
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
using MLJTuning
using StatisticalMeasures

function estimate_across_correlation_levels(estimators, σs; n=1000)
    results = Dict(key => [] for key in keys(estimators))
    for σ in σs
        dataset = generate_dataset(n=n, σ=σ)
        for (estimator_key, estimator) in estimators
            result, _ = estimator(Ψ, dataset; verbosity=0)
            push!(results[estimator_key], result)
        end
    end
    return results
end

function estimate_across_sample_sizes_and_correlation_levels(estimators, ns, σs)
    results = []
    for n in ns
        results_at_n = estimate_across_correlation_levels(estimators, σs; n=n)
        push!(results, results_at_n)
    end
    return results
end

function plot_across_sample_sizes_and_correlation_levels(results, ns, σs; estimator="TMLE_SL", title="Estimation via TMLE (GLMs)")
    fig = Figure(size=(800, 800))
    for (index, n) in enumerate(ns)
        results_at_n = results[index][estimator]
        Ψ̂s = TMLE.estimate.(results_at_n)
        errors = last.(confint.(significance_test.(results_at_n))) .- Ψ̂s
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

Random.seed!(123)

μT(w) = [sum(w), sum(w)]

μY(t, w) = 1 + 10t[1] - 3t[2] * t[1] * w
````

We assume that `W1` and `W2`, the confounding variables, follow a uniform distribution.

````@example interactions_correlated
generate_W(n) = rand(Uniform(0, 1), 2, n)
````

`T1` and `T2` are generated via a copula method through a multivariate normal to induce some statistical dependence (via σ).

````@example interactions_correlated
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
````

Finally, `Y` is generated through a simple linear model with an interaction term.

````@example interactions_correlated
function generate_Y(T, W1, n; σY=1)
    Y = zeros(Float64, n)
    for i in 1:n
        dY = Normal(μY(T[:, i], W1[i]), σY)
        Y[i] = rand(dY)
    end
    return Y
end
````

Importantly, the average interaction effect between `T1` and `T2` is thus ``-3 \mathbb{E}[W] = -1.5``.

We will generate a full dataset with the following function.

````@example interactions_correlated
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
````

Let's verify that each treatment level is sufficiently present in the dataset (≈positivity).

````@example interactions_correlated
combine(groupby(dataset, [:T1, :T2]), proprow => :JOINT_TREATMENT_FREQ)
````

And that `T1` and `T2` are indeed correlated.

````@example interactions_correlated
treatment_correlation(dataset) = cor(unwrap.(dataset.T1), unwrap.(dataset.T2))
@assert treatment_correlation(dataset) > 0.2 #hide
treatment_correlation(dataset)
````

## Estimation

We can now proceed to estimation, for instance using TMLE with linear models.

First, let's define the effect of interest. Interactions are defined via the `AIE` function (note that we only set `W1` as a confounder).

````@example interactions_correlated
Ψ = AIE(
    outcome=:Y,
    treatment_values= (
        T1=(case=1, control=0),
        T2=(case=1, control=0)
    ),
    treatment_confounders = [:W1]
)
linear_models = default_models(G=LogisticClassifier(lambda=0), Q_continuous=LinearRegressor())
estimator = Tmle(models=linear_models, weighted=true)
result, _ = estimator(Ψ, dataset; verbosity=0)
@assert pvalue(significance_test(result, -1.5)) > 0.05 #hide
significance_test(result)
````

The true effect size is thus covered by our confidence interval.

## Varying levels of correlation

We will now vary the correlation level between `T1` and `T2` to observe how it affects the estimation results across samples sizes. We will also
look at three different modelling strategies:
1. Generalized linear models (GLMs)
2. XGBoost
3. Super Learning (SL) via a model selection approach

First, let's see how the parameter σ affects the correlation between `T1` and `T2`.

````@example interactions_correlated
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
````

As expected, the correlation between `T1` and `T2` increases with σ. Let's see how this affects estimation.

We first define our Super Learners, they compare L2 penalized GLM and XGboost models for various penalization parameters λ
on a holdout set and select the best model.

````@example interactions_correlated
lambdas = 10 .^ range(1, stop=-4, length=5)
linear_regressors = [RidgeRegressor(lambda=λ) for λ in lambdas]
logistic_classifiers = [LogisticClassifier(lambda=λ) for λ in lambdas]
xgboost_classifiers = [XGBoostClassifier(tree_method="hist", lambda=λ, nthread=1) for λ in lambdas]
xgboost_regressors = [XGBoostRegressor(tree_method="hist", lambda=λ, nthread=1) for λ in lambdas]

sl_regressor = TunedModel(
    models=vcat(linear_regressors, xgboost_regressors),
    resampling=Holdout(),
    measure=rmse,
    check_measure=false
)

sl_classifier = TunedModel(
    models=vcat(logistic_classifiers, xgboost_classifiers),
    resampling=Holdout(),
    measure=log_loss,
    check_measure=false
)
````

Now define the sample sizes and correlation levels we want to explore.

````@example interactions_correlated
ns = [1000, 10_000, 100_000, 500_000]
σs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999]
````

and estimate (this will take a little while).

````@example interactions_correlated
estimators = Dict(
    "TMLE_GLM"  => Tmle(models=linear_models, weighted=true),
    "TMLE_XGBOOST" => Tmle(
        models=default_models(G=XGBoostClassifier(tree_method="hist", nthread=1), Q_continuous=XGBoostRegressor(tree_method="hist", nthread=1)),
        weighted=true,
    ),
    "TMLE_SL" => Tmle(
        models=default_models(G=sl_classifier, Q_continuous=sl_regressor),
        weighted=true,
    )
)

results = estimate_across_sample_sizes_and_correlation_levels(estimators, ns, σs)
````

Let us first focus on results obtained with the GLM estimator. In small sample sizes, coverage is almost perfect across all correlation levels. However, as sample size increases,
the confidence intervals shrink and start to miss the ground truth. The phenomenon is more pronounced for larger correlations. This could be due to model misspecification bias
which can be verified by using a more flexible modelling strategy, here we use XGBoost.

````@example interactions_correlated
plot_across_sample_sizes_and_correlation_levels(results, ns, σs; estimator="TMLE_GLM", title="Estimation via TMLE (GLMs)")
````

As expected, XGBoost improves estimation performance in the asymptotic regime, however,
the performance is the small sample size regime is deteriorated, likely due to over-fitting. To find the sweet spot between GLM and XGBoost,
we can resort to model selection to adaptively select the best model (sometimes this is called discrete super learning).

````@example interactions_correlated
plot_across_sample_sizes_and_correlation_levels(results, ns, σs; estimator="TMLE_XGBOOST", title="Estimation via TMLE (XGBoost)")
````

As we can see, the performance is now good across all sample sizes. Furthermore, the correlation between `T1` and `T2` seems harmless except when σ > 0.9.
The confidence interval is then quite large which will result in a loss of power.

````@example interactions_correlated
plot_across_sample_sizes_and_correlation_levels(results, ns, σs; estimator="TMLE_SL", title="Estimation via TMLE (SL)")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

