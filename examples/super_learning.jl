#=
# Becoming a Super Learner

## What this tutorial is about

Super Learning, also known as Stacking, is an ensemble technique that was first introduced by Wolpert in 1992. 
Instead of selecting a model based on cross-validation performance, models are combined by a meta-learner 
to minimize the cross-validation error. It has also been shown by van der Laan et al. that the resulting 
Super Learner will perform at least as well as its best performing submodel.

Why is it important for Targeted Learning?

The short answer is that the consistency (convergence in probability) of the targeted 
estimator depends on the consistency of at least one of the nuisance estimands: ``Q_0`` or ``G_0``.
By only using unrealistic models like linear models, we have little chance of satisfying the above criterion.
Super Learning is a data driven way to leverage a diverse set of models and build the best performing 
estimator for both ``Q_0`` or ``G_0``.

In the following, we investigate the benefits of Super Learning for the estimation of the Average Treatment Effect.

## The dataset

For this example we will use the following perinatal dataset. The (haz01, parity01) are converted to 
categorical values.
=#
using CSV
using DataFrames
using TMLE
using CairoMakie
using MLJ

dataset = CSV.read(
    joinpath(pkgdir(TMLE), "test", "data", "perinatal.csv"),
    DataFrame,
    select=[:haz01, :parity01, :apgar1, :apgar5, :gagebrth, :mage, :meducyrs, :sexn],
    types=Float64
)
dataset.haz01 = categorical(dataset.haz01)
dataset.parity01 = categorical(dataset.parity01)
nothing # hide

#=

We will also assume the following causal model:

=#

scm = SCM(
    SE(:haz01, [:parity01, :apgar1, :apgar5, :gagebrth, :mage, :meducyrs, :sexn]),
    SE(:parity01, [:apgar1, :apgar5, :gagebrth, :mage, :meducyrs, :sexn])
)

#=

## Defining a Super Learner in MLJ

In MLJ, a Super Learner can be defined using the [Stack](https://alan-turing-institute.github.io/MLJ.jl/stable/model_stacking/)
function. The three most important type of arguments for a Stack are:
- `metalearner`: The metalearner to be used to combine the weak learner to be defined.
- `resampling`: The cross-validation scheme, by default, a 6-fold cross-validation.
- `models...`: A series of named MLJ models.

One important point is that MLJ does not provide any model by itself, those have to be loaded from 
external compatible libraries. You can search for available models that match your data.

In our case, for both ``G_0`` and ``Q_0`` we need classification models and we can see there are quire a few of them:
=#

G_available_models = models(matching(dataset[!, parents(scm.parity01)], dataset.parity01))

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

function superlearner_models()
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
    
superlearner = Stack(;
    metalearner = LogisticClassifier(lambda=0),
    resampling  = StratifiedCV(nfolds=3),
    measure     = log_loss,
    superlearner_models()...
)

nothing # hide

#=

and assign those models to the `SCM`:

=#

setmodel!(scm.haz01, with_encoder(superlearner))
setmodel!(scm.parity01, superlearner)

#=

## Targeted estimation

Let us move to the targeted estimation step itself. We define the target estimand (the ATE):
=#
Ψ = ATE(
    scm,
    outcome=:haz01,
    treatment=(parity01=(case=true, control=false),),
)


nothing # hide

#=
Finally run the TMLE procedure:
=#
tmle_result, _ = tmle!(Ψ, dataset)

tmle_result

#
using Test # hide
pvalue(test_result) > 0.05 #hide
nothing # hide


