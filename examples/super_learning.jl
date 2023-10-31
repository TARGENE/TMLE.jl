#=
# Becoming a Super Learner

## What this tutorial is about

Super Learning, also known as Stacking, is an ensemble technique that was first introduced by Wolpert in 1992. 
Instead of selecting a model based on cross-validation performance, models are combined by a meta-learner 
to minimize the cross-validation error. It has also been shown by van der Laan et al. that the resulting 
Super Learner will perform at least as well as its best performing submodel (at least asymptotically).

Why is it important for Targeted Learning?

The short answer is that the consistency (convergence in probability) of the targeted 
estimator depends on the consistency of at least one of the nuisance estimands: ``Q_0`` or ``G_0``.
By only using unrealistic models like linear models, we have little chance of satisfying the above criterion.
Super Learning is a data driven way to leverage a diverse set of models and build the best performing 
estimator for both ``Q_0`` or ``G_0``.

## The dataset

Let's consider the case where Y is categorical. In TMLE.jl, this could be useful to learn:

- The propensity score
- The outcome model when the outcome is binary

We will use the following moons dataset:
=#
using MLJ

X, y = MLJ.make_moons(1000)
nothing # hide

#=

## Defining a Super Learner in MLJ

In MLJ, a Super Learner can be defined using the [Stack](https://alan-turing-institute.github.io/MLJ.jl/stable/model_stacking/)
function. The three most important type of arguments for a Stack are:
- `metalearner`: The metalearner to be used to combine the weak learner to be defined. Typically a generalized linear model.
- `resampling`: The cross-validation scheme, by default, a 6-fold cross-validation. Since we are working with categorical 
data it is a good idea to make sure the splits are balanced. We will thus use a `StratifiedCV` resampling strategy.
- `models...`: A series of named MLJ models.

One important point is that MLJ does not provide any model by itself, juat the API, models have to be loaded from 
external compatible libraries. You can search for available models that match your data.

=#

models(matching(X, y))

#=
!!! note "Stack limitation"
    The Stack cannot contain `<:Deterministic` models for classification.

Let's load a few packages providing models and build our first Stack:
=#

using MLJXGBoostInterface
using MLJLinearModels
using NearestNeighborModels

resampling = StratifiedCV()
metalearner = LogisticClassifier()

stack = Stack(
    metalearner = metalearner,
    resampling  = resampling,
    lr          = LogisticClassifier(),
    knn         = KNNClassifier(K=3)
)

#=
This Stack only contains 2 different models: a logistic classifier and a KNN classifier.
A Stack is just like any MLJ model, it can be wrapped in a `machine` and fitted:
=#

mach = machine(stack, X, y)
fit!(mach, verbosity=0)

#=
Or evaluated. Because the Stack contains a cross-validation procedure, this will result in two nested levels of resampling.
=#

evaluate!(mach, measure=log_loss, resampling=resampling)

#=

## A more advanced Stack

What are good Stack members? Virtually anything, provided they are MLJ models. Here are a few examples:

- You can use the stack to "select" model hyper-parameters. e.g. `KNNClassifier(K=3)` or `KNNClassifier(K=2)`?
- You can also use [self-tuning models](https://alan-turing-institute.github.io/MLJ.jl/stable/tuning_models/). Note that because 
these models resort to cross-validation, fitting the stack will result in two nested 
levels of sample-splitting.

The following self-tuned XGBoost will vary some hyperparameters in an internal sample-splitting procedure in order to optimize the Log-Loss. 
It will then be combined with the rest of the models in the Stack's own sample-splitting procedure. Finally, evaluation is performed in an outer sample-split.
=#

xgboost = XGBoostClassifier(tree_method="hist")
self_tuning_xgboost = TunedModel(
    model = xgboost,
    resampling = resampling,
    tuning = Grid(goal=20),
    range = [
        range(xgboost, :max_depth, lower=3, upper=7), 
        range(xgboost, :lambda, lower=1e-5, upper=10, scale=:log)
        ],
    measure = log_loss,
)

stack = Stack(
    metalearner         = metalearner,
    resampling          = resampling,
    self_tuning_xgboost = self_tuning_xgboost,
    lr                  = LogisticClassifier(),
    knn_2               = KNNClassifier(K=2),
    knn_3               = KNNClassifier(K=3)
)

mach = machine(stack, X, y)
evaluate!(mach, measure=log_loss, resampling=resampling)

#=

## Diagnostic

Optionally, one can also investigate how sucessful the weak learners were in the Stack's internal 
cross-validation. This is done by specifying the `measures` keyword argument. 

Here we look at both the Log-Loss and the AUC.

=#

stack.measures = [log_loss, auc]
fit!(mach, verbosity=0)
report(mach).cv_report

#=

One can look at the fitted parameters for the metalearner as well:
=#

fitted_params(mach).metalearner

