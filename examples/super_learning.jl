#=
# Becoming a Super Learner

## What this tutorial is about

Super Learning, also known as Stacking, is an emsemble technique that was first introduced by Wolpert in 1992. 
Instead of selecting a model based on cross-validation performance, models are combined by a meta-learner 
to minimize the cross-validation error. It has also been shown by van der Laan et al. that the resulting 
Super Learner will perform at least as well as its best performing submodel.

Why is it important for Targeted Learning?

The short answer is that the consistency (convergence in probability) of the targeted 
estimator depends on the consistency of at least one of the nuisance estimands: ``Q_0`` or ``G_0`` (see [Mathematical setting](@ref) ).
By only using unrealistic models like linear models, we have little chance of satisfying the above criterion.
Super Learning is a data driven way to leverage a diverse set of models and build the best performing 
estimator for both ``Q_0`` or ``G_0``.

In the following, we investigate the benefits of Super Learning for the estimation of [The Average Treatment Effect](@ref).

## The dataset

To demonstrate the benefits of Super Learning, we need to simulate a dataset more stimulating
than a simple linear regression, otherwise there would be nothing to do. Here is what I could 
come up with:
=#

using DataFrames
using TMLE
using Random
using Distributions
using StableRNGs
using CairoMakie
using CategoricalArrays
using MLJ
using TMLE
using LogExpFunctions

μY(T, W₁, W₂) = sin.(10T.*W₁).*exp.(-1 .+ 2W₁.*W₂ .- T.*W₂) .- cos.(10T.*W₂).*log.(2 .- W₁.*W₂)
μT(W₁, W₂) = logistic.(10sin.(W₁) .- 1.5W₂)

function hard_problem(;n=1000, doT=nothing)
    rng = StableRNG(123)
    W₁ = rand(rng, Uniform(), n)
    W₂ = rand(rng, Uniform(), n)
    T = doT === nothing ?
        rand(rng, Uniform(), n) .< μT(W₁, W₂) :
        repeat([doT], n)
    Y = μY(T, W₁, W₂) .+ rand(rng, Normal(0, 0.1), n)
    return DataFrame(W₁=W₁, W₂=W₂, T=categorical(T), Y=Y)
end

N = 10000
dataset = hard_problem(;n=N)
nothing # hide

#=
It may seem difficult to understand it but we can still have a look at the function ``\mathbf{E}[Y|T,W₁, W₂]``.
=#


function plot_μY(;npoints=100)
    W₁ = LinRange(0, 1, npoints)
    W₂ = LinRange(0, 1, npoints)
    y₁ = [μY(1, w₁, w₂) for w₁ in W₁, w₂ in W₂]
    y₀ = [μY(0, w₁, w₂) for w₁ in W₁, w₂ in W₂]
    t = [μT(w₁, w₂) for w₁ in W₁, w₂ in W₂]
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), resolution = (1000, 700))
    ax₁ = Axis3(fig[1, 1], aspect=:data, xlabel="W₁", ylabel="W₂", zlabel="μY")
    Label(fig[1, 1, Top()], "μY(W₁, W₂, T=1)", fontsize = 30, tellwidth=false, tellheight=false)
    surface!(ax₁, W₁, W₂, y₁)
    ax₀ = Axis3(fig[1, 2], aspect=:data, xlabel="W₁", ylabel="W₂", zlabel="μY")
    Label(fig[1, 2, Top()], "μY(W₁, W₂, T=0)", fontsize = 30, tellwidth=false, tellheight=false)
    surface!(ax₀, W₁, W₂, y₀)
    axₜ = Axis3(fig[2, :], aspect=:data, xlabel="W₁", ylabel="W₂", zlabel="μT")
    Label(fig[2, :, Top()], "μT(W₁, W₂)", fontsize = 30, tellwidth=false, tellheight=false)
    surface!(axₜ, W₁, W₂, t)
    return fig
end

plot_μY(;npoints=100)

#=
Also, even though the analytical solution for the ATE may be intractable, since we known the
data generating process, we can approximate it by sampling.
=#

function approximate_ate(;n=1000)
    dataset_1 = hard_problem(;n=n, doT = 1)
    dataset_0 = hard_problem(;n=n, doT = 0)
    return mean(dataset_1.Y) - mean(dataset_0.Y)
end

ψ₀ =  approximate_ate(;n=1000)

#=
Now that the data has been generated and we know the solution to our problem, we can dive into the estimation
part.

## Defining a Super Learner in MLJ

In MLJ, a Super Learner can be defined using the [Stack](https://alan-turing-institute.github.io/MLJ.jl/stable/model_stacking/)
function. The three most important type of arguments for a Stack are:
- `metalearner`: The metalearner to be used to combine the weak learner to be defined.
- `resampling`: The cross-validation scheme, by default, a 6-fold cross-validation.
- `models...`: A series of named MLJ models.

One important point is that MLJ does not provide any model by itself, those have to be loaded from 
external compatible libraries. You can search for available models that match your data.

for instance, for ``G_0 = P(T| W_1, W_2)``, we need classification models and we can see there are quire a bunch of them:
=#

G_available_models = models(matching(dataset[!, [:W₁, :W₂]], dataset.T))

#=
!!! note "Stack limitations"
    For now, there are a few limitations as to which models you can actually use within the Stack.
    The most important is that if the output is categorical, each model must be `<: Probabilistic`,
    which means that SVMs cannot be used as a weak learners for classification.

Let's load a few model providing libraries and define our library for ``G_0``.
=#

using EvoTrees
using MLJLinearModels
using MLJModels
using NearestNeighborModels

function Gmodels()
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1., 10., 100.]
    logistic_models = [LogisticClassifier(lambda=l) for l in lambdas]
    logistic_models = NamedTuple{Tuple(Symbol("lr_$i") for i in eachindex(lambdas))}(logistic_models)
    evo_trees = [EvoTreeClassifier(lambda=l) for l in lambdas]
    evo_trees = NamedTuple{Tuple(Symbol("tree_$i") for i in eachindex(lambdas))}(evo_trees)
    Ks = [5, 10, 50, 100]
    knns = [KNNClassifier(K=k) for k in Ks]
    knns = NamedTuple{Tuple(Symbol("knn_$i") for i in eachindex(Ks))}(knns)
    return merge(logistic_models, evo_trees, knns)
end
    
G_super_learner = Stack(;
    metalearner = LogisticClassifier(lambda=0),
    resampling  = StratifiedCV(nfolds=3),
    measure     = log_loss,
    Gmodels()...
)


nothing # hide

#=
And now do the same for ``Q_0``
=#

function Qmodels()
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1., 10., 100.]
    linear_models = [RidgeRegressor(lambda=l) for l in lambdas]
    linear_models = NamedTuple{Tuple(Symbol("lr_$i") for i in eachindex(lambdas))}(linear_models)
    evo_trees = [EvoTreeRegressor(lambda=l) for l in lambdas]
    evo_trees = NamedTuple{Tuple(Symbol("tree_$i") for i in eachindex(lambdas))}(evo_trees)
    Ks = [5, 10, 50, 100]
    knns = [KNNRegressor(K=k) for k in Ks]
    knns = NamedTuple{Tuple(Symbol("knn_$i") for i in eachindex(Ks))}(knns)
    return merge(linear_models, evo_trees, knns)
end

Q_super_learner = Stack(;
    metalearner = LinearRegressor(fit_intercept=false),
    resampling=CV(nfolds=3),
    Qmodels()...
    )

nothing # hide

#=
## Targeted estimation

Let us move to the targeted estimation step itself. We define the target estimand (the ATE) and the nuisance estimands specification:
=#
Ψ = ATE(
    outcome = :Y,
    treatment = (T=(case=true, control=false),),
    confounders = [:W₁, :W₂]
)

η_spec = NuisanceSpec(
    Q_super_learner,
    G_super_learner
)

nothing # hide

#=
Finally run the TMLE procedure and check the result
=#
tmle_result, cache = tmle(Ψ, η_spec, dataset)

test_result = OneSampleTTest(tmle_result.tmle, ψ₀)

#=
Now, what if we had used linear models only instead of the Super Learner? This is easy to check
=#

η_spec_linear = NuisanceSpec(
    LinearRegressor(),
    LogisticClassifier(lambda=0)
)

tmle_result_linear, cache = tmle(Ψ, η_spec_linear, dataset)

test_result_linear = OneSampleTTest(tmle_result_linear.tmle, ψ₀)

#
using Test # hide
pvalue(test_result) > 0.05 #hide
nothing # hide


