# Resampling Strategies

In causal inference, the treatment variables play a crucial role, it is thus usually a good idea to stratify cross-validation schemes by these variables. This is the purpose of the following resampling strategy which is an instance of a `MLJBase.ResamplingStrategy`. It applies a stratified cross-validation strategy based on both treatments and outcome (if it is Finite) variables.

```julia
resampling = CausalStratifiedCV(resampling=StratifiedCV(nfolds=3))
```

For ease of use the treatment variables are detected automatically from the parameter of interest ``Ψ``.