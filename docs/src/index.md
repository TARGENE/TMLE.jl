```@meta
CurrentModule = TMLE
```

# TMLE

The purpose of this package is to provide convenience methods for 
Targeted Minimum Loss-Based Estimation (TMLE). TMLE is a framework for
efficient estimation that was first proposed by Van der Laan et al.
If you are new to TMLE, this [review paper](https://www.hindawi.com/journals/as/2014/502678/) 
gives a nice overview to the field. Because TMLE requires nuisance parameters 
to be learnt by machine learning algorithms, this package is built on top of 
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/). This means that any model 
respecting the MLJ interface can be used to estimate the nuisance parameters.

## Installation

The package is not yet part of the registry and must be installed via github:

```julia
julia> ]add https://github.com/olivierlabayle/TMLE.jl
```

```@index
```

```@autodocs
Modules = [TMLE]
```
