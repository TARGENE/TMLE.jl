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

allcolumns(Ψ::Parameter) = vcat(confounders_and_covariates(Ψ), treatments(Ψ), target(Ψ))
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

struct NuisanceSpec
    Q::MLJBase.Model
    G::MLJBase.Model
    H::MLJBase.Model
    F::MLJBase.Model
end

"""
    NuisanceSpec(Q, G; H=encoder(), F=Q_model(target_scitype(Q)))

Specification of the nuisance parameters to be learnt.

# Arguments:

- Q: For the estimation of E₀[Y|T=case, X]
- G: For the estimation of P₀(T|W)
- H: The `TreatmentTransformer`` to deal with categorical treatments
- F: The generalized linear model used to fluctuate the initial Q
"""
NuisanceSpec(Q, G; H=TreatmentTransformer(), F=Q_model(target_scitype(Q))) =
    NuisanceSpec(Q, G, H, F)

Q_model(::Type{<:AbstractVector{<:MLJBase.Continuous}}) =
    LinearRegressor(fit_intercept=false, offsetcol = :offset)

Q_model(::Type{<:AbstractVector{<:Finite}}) =
    LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)

Q_model(t::Type{Any}) = throw(ArgumentError("Cannot proceed with Q model with target_scitype $t"))

namedtuples_from_dicts(d) = d
namedtuples_from_dicts(d::Dict) = 
    NamedTuple{Tuple(keys(d))}([namedtuples_from_dicts(val) for val in values(d)])


"""
    parameters_from_yaml(path)

Instantiate parameters described in the provided YAML file.
"""
function parameters_from_yaml(path)
    config = YAML.load_file(path; dicttype=Dict{Symbol,Any})
    parameters = Parameter[]
    W = Symbol.(config[:W])
    C = haskey(config, :C) ? Symbol.(config[:C]) : []
    Ys = Symbol.(config[:Y])
    for param_entry in config[:Parameters]
        param_string = pop!(param_entry, :name)
        paramtype = getfield(TMLE, Symbol(param_string))
        T = namedtuples_from_dicts(param_entry)
        for Y in Ys
            push!(parameters, paramtype(;target=Y, treatment=T, confounders=W, covariates=C))
        end
    end
    return parameters
end