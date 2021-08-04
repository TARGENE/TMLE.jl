```@meta
CurrentModule = TMLE
```

# TMLE

The purpose of this package is to provide convenience methods for 
Targeted Minimum Loss-Based Estimation (TMLE). TMLE is a framework for
efficient estimation that was first proposed by Van der Laan et al in 2006.
If you are new to TMLE, this [review paper](https://www.hindawi.com/journals/as/2014/502678/) 
gives a nice overview to the field. Because TMLE requires nuisance parameters 
to be learnt by machine learning algorithms, this package is built on top of 
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). This means that any model 
respecting the MLJ interface can be used to estimate the nuisance parameters.

!!! note 
    This package is still experimental and documentation under construction


## Installation

The package is not yet part of the registry and must be installed via github:

```julia
julia> ]add https://github.com/olivierlabayle/TMLE.jl
```

## Get in touch

Please feel free to fill an issue if you want to report any bug
or want to have additional features part of the package. 
Contributing is also welcome.

## Tutorials

This package is built on top of MLJ, if you are new to the MLJ framework, 
please refer first to their [documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/).

Currently, two parameters of the generating distribution are available for estimation, the
Average Treatment Effect (ATE) and the Interaction Average 
Treatment Effect (IATE). For both quantities, a graphical representation of the 
underlying causal model in presented bellow.

![causal_model.png](img/causal_model.png)

TMLE is a two steps procedure, it first starts by estimating nuisance 
parameters that will be used to build the final estimator. They are called nuisance parameters
because they are required for estimation but are not our target quantity of interest. 
For both the ATE and IATE, the nuisance parameters that require a learning algorithm are:

- The conditional extectation of the target 
- The conditional density of the treatment

They are typically estimated by stacking which is built into MLJ
and you can find more information about it [here](https://alan-turing-institute.github.io/MLJ.jl/dev/composing_models/#Model-Stacking). Stacking is not compulsory however and any model 
respecting the [MLJ Interface](https://github.com/JuliaAI/MLJModelInterface.jl) should work out of the box.

In the second stage, TMLE fluctuates a nuisance parameter using a parametric model in order to
solve the efficient influence curve equation. For now, this is done via a 
Generalized Linear model and the nuisance parameter which is fluctuated is the
conditional extectation of the target variable.

For those examples, we will need the following packages:

```julia
using Random
using Distributions
using MLJ
using TMLE
```

### ATE

Let's consider the following example:

- W = [W_1, W_2, W_3] is a set of binary confounding variables, W ~ Bernoulli(0.5)
- T is a Binary variable, p(T=1|W=w) = expit(0.5W_1_ + 1.5W_2 - W_3)
- Y is a Continuous variable, Y = T + 2W_1 + 3W_2 - 4W_3 + \epsilon(0, 1)

```julia
n = 10000
rng = MersenneTwister(0)
# Sampling
Unif = Uniform(0, 1)
W = float(rand(rng, Bernoulli(0.5), n, 3))
t = rand(rng, Unif, n) .< expit(0.5W[:, 1] + 1.5W[:, 2] - W[:,3])
y = t + 2W[:, 1] + 3W[:, 2] - 4W[:, 3] + rand(rng, Normal(0, 1), n)
# W needs to respect the Tables.jl interface.
# t is a binary categorical vector
W = MLJ.table(W)
t = categorical(t)
```

We need to define 2 estimators for the nuisance parameters, usually this is 
done using the Stack but here because we know the generating process we can 
cheat a bit. We will use a Logistic Classifier for p(T|W) and a Constant Regressor
for p(Y|W, T). This means one estimator is well specified and the other not. 
The target is continuous thus we will use a Linear regression model 
for the fluctuation. This is done by specifying a Normal distribution for the 
Generalized Linear Model.

```julia

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

tmle = ATEEstimator(LogisticClassifier(),
                    MLJ.DeterministicConstantRegressor(),
                    Normal())

```

Now, all there is to do is to fit the estimator:

```julia
fitresult, _, _ = MLJ.fit(tmle, 0, t, W, y)
```

The `fitresult` contains the estimate and the associated standard error.

### IATE

TODO.

## API 

```@autodocs
Modules = [TMLE]
Private = false
```


```@index
```