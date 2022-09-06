const causal_graph = """
     T  ←  W  
      ↘   ↙ 
        Y  ← C

## Notation:

- Y: target
- T: treatment
- W: confounders
- C: covariates
- X = (W, C, T) 
"""

"""
A Parameter is a functional on distribution space Ψ: ℳ → ℜ. 
"""
abstract type Parameter end

"""
# CM: Conditional Mean

Mathematical definition: 

    Eₓ[E[Y|T=t, X]]

# Causal graph:

$causal_graph

# Fields:
    - target: A symbol identifying the target variable of interest
    - treatment: A NamedTuple linking each treatment variable to a value
    - confounders: Confounding variables affecting both the target and the treatment
    - covariates: Optional extra variables affecting the target only

# Examples:
```julia
CM₁ = CM(
    target=:Y₁,
    treatment=(T₁=1,),
    confounders=[:W₁, :W₂],
    covariates=[:C₁]
)

CM₂ = CM(
    target=:Y₂,
    treatment=(T₁=1, T₂="A"),
    confounders=[:W₁],
)
```
"""
struct CM <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::AbstractVector{Symbol}
    covariates::AbstractVector{Symbol}
end

CM(;target, treatment, confounders, covariates=[]) = 
    CM(target, treatment, confounders, covariates)

"""
# ATE: Average Treatment Effect

Mathematical definition: 

    Eₓ[E[Y|T=case, X]] - Eₓ[E[Y|T=control, X]]

# Causal graph:

$causal_graph

# Fields:
    - target: A symbol identifying the target variable of interest
    - treatment: A NamedTuple linking each treatment variable to case/control values
    - confounders: Confounding variables affecting both the target and the treatment
    - covariates: Optional extra variables affecting the target only

# Examples:
```julia
ATE₁ = ATE(
    target=:Y₁,
    treatment=(T₁=(case=1, control=0),),
    confounders=[:W₁, :W₂],
    covariates=[:C₁]
)

ATE₂ = ATE(
    target=:Y₂,
    treatment=(T₁=(case=1, control=0), T₂=(case="A", control="B")),
    confounders=[:W₁],
)
```
"""
struct ATE <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::AbstractVector{Symbol}
    covariates::AbstractVector{Symbol}
end

ATE(;target, treatment, confounders, covariates=[]) = 
    ATE(target, treatment, confounders, covariates)

"""
# IATE: Interaction Average Treatment Effect

Mathematical definition for pairwise interaction:

    Eₓ[E[Y|T₁=1, T₂=1, X]] - Eₓ[E[Y|T₁=1, T₂=0, X]] - Eₓ[E[Y|T₁=0, T₂=1, X]] + Eₓ[E[Y|T₁=0, T₂=0, X]]

# Causal graph:

$causal_graph

# Fields:
    - target: A symbol identifying the target variable of interest
    - treatment: A NamedTuple linking each treatment variable to case/control values
    - confounders: Confounding variables affecting both the target and the treatment
    - covariates: Optional extra variables affecting the target only

# Examples:
```julia
IATE₁ = IATE(
    target=:Y₁,
    treatment=(T₁=(case=1, control=0), T₂=(case="A", control="B")),
    confounders=[:W₁],
)
```
"""
struct IATE <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::AbstractVector{Symbol}
    covariates::AbstractVector{Symbol}
end

IATE(;target, treatment, confounders, covariates=[]) = 
    IATE(target, treatment, confounders, covariates)

selectcols(data, cols) = data |> TableOperations.select(cols...) |> Tables.columntable

confounders(Ψ::Parameter) = Ψ.confounders
confounders(dataset, Ψ) = selectcols(dataset, confounders(Ψ))

covariates(Ψ::Parameter) = Ψ.covariates
covariates(dataset, Ψ) = selectcols(dataset, covariates(Ψ))

treatments(Ψ::Parameter) = collect(keys(Ψ.treatment))
treatments(dataset, Ψ) = selectcols(dataset, treatments(Ψ))

target(Ψ::Parameter) = Ψ.target
target(dataset, Ψ) = Tables.getcolumn(dataset, target(Ψ))

treatment_and_confounders(Ψ::Parameter) = vcat(confounders(Ψ), treatments(Ψ))

confounders_and_covariates(Ψ::Parameter) = vcat(confounders(Ψ), covariates(Ψ))
confounders_and_covariates(dataset, Ψ) = selectcols(dataset, confounders_and_covariates(Ψ))

Qinputs(dataset, Ψ::Parameter) = 
    selectcols(dataset, vcat(confounders_and_covariates(Ψ), treatments(Ψ)))


"""
# NuisanceParameters

The set of estimators that need to be estimated but are not of direct interest.

# Causal graph:

$causal_graph

# Fields:

All fields are MLJBase.Machine.

    - Q: An estimator of E[Y|X]
    - G: An estimator of P(T|W)
    - H: A one-hot-encoder categorical treatments
    - F: A generalized linear model to fluctuate E[Y|X]
"""
mutable struct NuisanceParameters
    Q::Union{Nothing, MLJBase.Machine}
    G::Union{Nothing, MLJBase.Machine}
    H::Union{Nothing, MLJBase.Machine}
    F::Union{Nothing, MLJBase.Machine}
end


encoder(Ψ::Parameter) = OneHotEncoder(features=treatments(Ψ), drop_last=true, ordered_factor=false)

"""
    fit!(η::NuisanceParameters, η_spec, Ψ::Parameter, dataset; verbosity=1)

Fits the nuisance parameters η on the dataset using the specifications from η_spec
and the variables defined by Ψ.
"""
function fit!(η::NuisanceParameters, η_spec, Ψ::Parameter, dataset; verbosity=1)
    # Fitting P(T|W)
    # Only rows with missing values in either W or Tₜ are removed
    if η.G === nothing
        log_fit(verbosity, "P(T|W)")
        nomissing_WT = nomissing(dataset[:source], treatment_and_confounders(Ψ))
        W = confounders(nomissing_WT, Ψ)
        T = treatments(nomissing_WT, Ψ)
        mach = machine(adapt(η_spec.G, T), W, adapt(T))
        MLJBase.fit!(mach, verbosity=verbosity-1)
        η.G = mach
    else
        log_no_fit(verbosity, "P(T|W)")
    end

    # Fitting E[Y|X]
    if η.Q === nothing
        # Data
        X = Qinputs(dataset[:no_missing], Ψ)
        y = target(dataset[:no_missing], Ψ)

        # Fitting the Encoder
        if η.H === nothing
            log_fit(verbosity, "Encoder")
            mach = machine(encoder(Ψ), X)
            MLJBase.fit!(mach, verbosity=verbosity-1)
            η.H = mach
        else
            log_no_fit(verbosity, "Encoder")
        end
        log_fit(verbosity, "E[Y|X]")
        mach = machine(η_spec.Q, MLJBase.transform(η.H, X), y)
        MLJBase.fit!(mach, verbosity=verbosity-1)
        η.Q = mach
    else
        log_no_fit(verbosity, "Encoder")
        log_no_fit(verbosity, "E[Y|X]")
    end
end

function fluctuation_input(dataset, η, Ψ; threshold=1e-8)
    X = Qinputs(dataset, Ψ)
    offset = compute_offset(η.Q, MLJBase.transform(η.H, X))
    indicators = indicator_fns(Ψ)
    W = confounders(X, Ψ)
    T = treatments(X, Ψ)
    covariate = compute_covariate(η.G, W, T, indicators; 
                                        threshold=threshold)
    return fluctuation_input(covariate, offset)
end

function tmle!(η::NuisanceParameters, Ψ, dataset; verbosity=1, threshold=1e-8)
    X = fluctuation_input(dataset, η, Ψ, threshold=threshold)
    y = target(dataset, Ψ)
    mach = machine(TMLE.fluctuation_model(η.Q.model), X, y)
    MLJBase.fit!(mach, verbosity=verbosity-1)
    η.F = mach
end

function fluctuation_model(Q)
    if Q isa Probabilistic
        return LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)
    elseif Q isa Deterministic
        return LinearRegressor(fit_intercept=false, offsetcol = :offset)
    else
        throw(ArgumentError("Q should be either Probabilistic or Deterministic"))
    end
end

function outcome_mean(η, Ψ, dataset; threshold=1e-8)
    if η.F isa Nothing
        X = Qinputs(dataset, Ψ)
        return expected_value(
            MLJBase.predict(η.Q,  MLJBase.transform(η.H, X)), 
            typeof(η.Q.model), 
            target_scitype(η.Q.model)
        )
    else
        X = fluctuation_input(dataset, η, Ψ, threshold=threshold)
        return predict_mean(η.F, X)
    end
end

function counterfactual_aggregate(Ψ, η, dataset; threshold=1e-8)
    WC = TMLE.confounders_and_covariates(dataset, Ψ)
    T = TMLE.treatments(dataset, Ψ)
    counterfactual_aggregate_ = zeros(nrows(T))
    for (vals, sign) in TMLE.indicator_fns(Ψ)
        Tc = TMLE.counterfactualTreatment(vals, T)
        Xc = Qinputs(merge(WC, Tc), Ψ)
        counterfactual_aggregate_ .+= sign.* TMLE.outcome_mean(η, Ψ, Xc, threshold=threshold)
    end
    return counterfactual_aggregate_
end

function gradient_W(Ψ::Parameter, η::NuisanceParameters, dataset; threshold=1e-8)
    counterfactual_aggregate_ = counterfactual_aggregate(Ψ, η, dataset; threshold=threshold)
    return counterfactual_aggregate_ .- mean(counterfactual_aggregate_)
end

function gradient_Y_X(Ψ::Parameter, η::NuisanceParameters, dataset; threshold=1e-8)
    indicators = TMLE.indicator_fns(Ψ)
    W = TMLE.confounders(dataset, Ψ)
    T = TMLE.treatments(dataset, Ψ)
    covariate = TMLE.compute_covariate(η.G, W, T, indicators; 
                                        threshold=threshold)
    return covariate .* (float(TMLE.target(dataset, Ψ)) .- outcome_mean(η, Ψ, dataset, threshold=threshold))
end

gradient(Ψ::Parameter, η::NuisanceParameters, dataset; threshold=1e-8) = gradient_Y_X(Ψ, η, dataset; threshold=threshold) .+ gradient_W(Ψ, η, dataset; threshold=threshold)

estimate(Ψ, η, dataset; threshold=1e-8) = mean(counterfactual_aggregate(Ψ, η, dataset; threshold=threshold))
