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

```@raw html
<img src="assets/causal_model.png" alt="Causal Model" style="width:200px;"/>
```

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

expit(X) = 1 ./ (1 .+ exp.(-X))
```

### ATE

Let's consider the following example:

- W = [W_1, W_2, W_3] is a set of binary confounding variables, ``W \sim Bernoulli(0.5)``
- T is a Binary variable, ``p(T=1|W=w) = \text{expit}(0.5W_1 + 1.5W_2 - W_3)``
- Y is a Continuous variable, ``Y = T + 2W_1 + 3W_2 - 4W_3 + \epsilon(0, 1)``

For which the ATE can be computed explicitely and is equal to 1. In Julia such dataset
can be generated like this:

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

The `fitresult` contains the estimate and the associated standard error. We can see 
that even if one nuisance parameter is misspecified, the double robustness of TMLE
enables correct estimation of our target.


### IATE

The IATE measures the effect of interacting causes on a target variable, it was 
defined by Beentjes and Khamseh [in this paper](https://link.aps.org/doi/10.1103/PhysRevE.102.053314).
In this case, the treatment variable T is a vector, for instance for two treatments T=(T_1, T_2).

Let's consider the following example for which again the IATE is known:

- W is a binary outcome confounding variable, ``W \sim Bernoulli(0.4)``
- ``T =(T_1, T_2)`` are independent binary variables sampled from an expit model. ``p(T_1=1|W=w) = \text{expit}(0.5w - 1)`` and, ``p(T_2=1|W=w) = \text{expit}(-0.5w - 1)``
- Y is a binary variable sampled from an expit model. ``p(Y=1|t_1, t_2, w) = \text{expit}(-2w + 3t_1 - 3t_2 - 1)``

In Julia:

```julia
n = 10000
rng = MersenneTwister(0)
p_w() = 0.4
pt1_given_w(w) = expit(0.5w .- 1)
pt2_given_w(w) = expit(-0.5w .- 1)
py_given_t1t2w(t1, t2, w) = expit(-2w .+ 3t1 .- 3t2 .- 1)
# Sampling
Unif = Uniform(0, 1)
w = rand(rng, Unif, n) .< p_w()
t₁ = rand(rng, Unif, n) .< pt1_given_w(w)
t₂ = rand(rng, Unif, n) .< pt2_given_w(w)
y = rand(rng, Unif, n) .< py_given_t1t2w(t₁, t₂, w)
# W should be a table
# T should be a table of binary categorical variables
# Y should be a binary categorical variable
W = (W=convert(Array{Float64}, w),)
T = (t₁ = categorical(t₁), t₂ = categorical(t₂))
y = categorical(y)
# Compute the theoretical IATE
IATE₁ = (py_given_t1t2w(1, 1, 1) - py_given_t1t2w(1, 0, 1) - py_given_t1t2w(0, 1, 1) + py_given_t1t2w(0, 0, 1))*p_w()
IATE₀ = (py_given_t1t2w(1, 1, 0) - py_given_t1t2w(1, 0, 0) - py_given_t1t2w(0, 1, 0) + py_given_t1t2w(0, 0, 0))*(1 - p_w())
IATE = IATE₁ + IATE₀
```

Again, we need to estimate the 2 nuisance parameters, this time let's use the 
Stack with a few learning algorithms. The fluctuation will be a Logistic Regression,
this is done by specifying a Bernoulli distribution for the 
Generalized Linear Model.

```julia
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels verbosity=0

stack = Stack(;metalearner=LogisticClassifier(),
                resampling=CV(),
                lr=LogisticClassifier(),
                tree_2=DecisionTreeClassifier(max_depth=2),
                tree_3=DecisionTreeClassifier(max_depth=3),
                knn=KNNClassifier())

tmle = ATEEstimator(stack,
                    stack,
                    Bernoulli())
```

And fit it!

```julia
fitresult, _, _ = MLJ.fit(tmle, 0, t, W, y)
```


## API 


```@autodocs
Modules = [TMLE]
Private = false
```


```@index
```