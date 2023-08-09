# Miscellaneous

## Treatment Transformer

To account for the fact that treatment variables are categorical variables we provide a MLJ compliant transformer that will either:

- Retrieve the floating point representation of a treatment it it has a natural ordering
- One hot encode it otherwise

Such transformer can be created with:

```@example
using TMLE # hide
TreatmentTransformer(;encoder=encoder())
```

where `encoder` is a [OneHotEncoder](https://alan-turing-institute.github.io/MLJ.jl/dev/models/OneHotEncoder_MLJModels/#OneHotEncoder_MLJModels).

The `with_encoder(model; encoder=TreatmentTransformer())` provides a shorthand to combine a `TreatmentTransformer` with another MLJ model in a pipeline.

Of course you are also free to define your own strategy!