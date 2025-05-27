# Accelerations

Targeted estimators are expensive by nature, in particular when resorting to cross-validation and collaborative TMLE. However many of the operations can be conducted in parallel. To facilitate this, an `acceleration` parameter can be provided to the estimation call.

At the moment, only single CPU and multi-threading modes are supported. For a multi-threaded call:

```julia
tmle = Tmle(resampling=CausalStratifiedCV())
tmle(Ψ, dataset; acceleration=CPUThreads())
```

In this case, fitting across multiple folds will happen on all available threads. Similarly the outcome mean and propensity score model will be estimated in parallel.

!!! note
    As noted [here](https://juliaai.github.io/MLJ.jl/stable/acceleration_and_parallelism/), non native Julia MLJ models may not be suitable for multi-threading.