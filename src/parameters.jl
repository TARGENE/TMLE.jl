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

#####################################################################
###                    Abstract Parameter                         ###
#####################################################################
"""
A Parameter is a functional on distribution space Ψ: ℳ → ℜ. 
"""
abstract type Parameter end

#####################################################################
###                      Conditional Mean                         ###
#####################################################################

"""
# CM: Conditional Mean

Mathematical definition: 

    Eₓ[E[Y|do(T=t), X]]

# Assumed Causal graph:

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
@option struct CM <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::Vector{Symbol}
    covariates::Vector{Symbol} = Symbol[]
end

#####################################################################
###                  Average Treatment Effect                     ###
#####################################################################

"""
# ATE: Average Treatment Effect

Mathematical definition: 

    Eₓ[E[Y|do(T=case), X]] - Eₓ[E[Y|do(T=control), X]]

# Assumed Causal graph:

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
@option struct ATE <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::Vector{Symbol}
    covariates::Vector{Symbol} = Symbol[]
end


#####################################################################
###            Interaction Average Treatment Effect               ###
#####################################################################

"""
# IATE: Interaction Average Treatment Effect

Mathematical definition for pairwise interaction:

    Eₓ[E[Y|do(T₁=1, T₂=1), X]] - Eₓ[E[Y|do(T₁=1, T₂=0), X]] - Eₓ[E[Y|do(T₁=0, T₂=1), X]] + Eₓ[E[Y|do(T₁=0, T₂=0), X]]

# Assumed Causal graph:

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
@option struct IATE <: Parameter
    target::Symbol
    treatment::NamedTuple
    confounders::Vector{Symbol}
    covariates::Vector{Symbol} = Symbol[]
end


#####################################################################
###                     Nuisance Parameters                       ###
#####################################################################

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
    cache::Bool
end

"""
    NuisanceSpec(Q, G; H=encoder(), F=Q_model(target_scitype(Q)))

Specification of the nuisance parameters to be learnt.

# Arguments:

- Q: For the estimation of E₀[Y|T=case, X]
- G: For the estimation of P₀(T|W)
- H: The `TreatmentTransformer`` to deal with categorical treatments
- F: The generalized linear model used to fluctuate the initial Q
- cache: Whether corresponding machines will cache data or not.
"""
NuisanceSpec(Q, G; H=TreatmentTransformer(), F=F_model(target_scitype(Q)), cache=true) =
    NuisanceSpec(Q, G, H, F, cache)

#####################################################################
###                         Methods                               ###
#####################################################################

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

F_model(::Type{<:AbstractVector{<:MLJBase.Continuous}}) =
    LinearRegressor(fit_intercept=false, offsetcol = :offset)

F_model(::Type{<:AbstractVector{<:Finite}}) =
    LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)

F_model(t::Type{Any}) = throw(ArgumentError("Cannot proceed with Q model with target_scitype $t"))

namedtuples_from_dicts(d) = d
namedtuples_from_dicts(d::Dict) = 
    NamedTuple{Tuple(keys(d))}([namedtuples_from_dicts(val) for val in values(d)])


function param_key(Ψ::Parameter)
    return (
        join(Ψ.confounders, "_"),
        join(keys(Ψ.treatment), "_"),
        string(Ψ.target),
        join(Ψ.covariates, "_")
    )
end

"""
    optimize_ordering!(parameters::Vector{<:Parameter})

Reorders the given parameters so that most nuisance parameters fits can be
reused. Given the assumed causal graph:

$causal_graph

and the requirements to estimate both p(T|W) and E[Y|W, T, C].

A natural ordering of the parameters in order to save computations is given by the
following variables ordering: (W, T, Y, C)
"""
optimize_ordering!(parameters::Vector{<:Parameter}) = sort!(parameters, by=param_key)

"""
    optimize_ordering(parameters::Vector{<:Parameter})

See [`optimize_ordering!`](@ref)
"""
optimize_ordering(parameters::Vector{<:Parameter}) = sort(parameters, by=param_key)