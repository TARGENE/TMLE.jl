const causal_graph = """
     T  ←  W  
      ↘   ↙ 
        Y  ← C

## Notation:

- Y: outcome
- T: treatment
- W: confounders
- C: covariates
- X = (W, C, T) 
"""

#####################################################################
###                    Abstract Estimand                          ###
#####################################################################
"""
A Estimand is a functional on distribution space Ψ: ℳ → ℜ. 
"""
abstract type Estimand end

function MLJBase.fit!(Ψ::Estimand, dataset; verbosity=1, cache=true, force=false)
    for eq in equations_to_fit(Ψ)
        fit!(eq, dataset, verbosity=verbosity, cache=cache, force=force)
    end
end

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
    - outcome: A symbol identifying the outcome variable of interest
    - treatment: A NamedTuple linking each treatment variable to a value
    - confounders: Confounding variables affecting both the outcome and the treatment
    - covariates: Optional extra variables affecting the outcome only

# Examples:
```julia
CM₁ = CM(
    outcome=:Y₁,
    treatment=(T₁=1,),
    confounders=[:W₁, :W₂],
    covariates=[:C₁]
)

CM₂ = CM(
    outcome=:Y₂,
    treatment=(T₁=1, T₂="A"),
    confounders=[:W₁],
)
```
"""
@option struct CM <: Estimand
    scm::StructuralCausalModel
    outcome::Symbol
    treatment::NamedTuple
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
    - outcome: A symbol identifying the outcome variable of interest
    - treatment: A NamedTuple linking each treatment variable to case/control values
    - confounders: Confounding variables affecting both the outcome and the treatment
    - covariates: Optional extra variables affecting the outcome only

# Examples:
```julia
ATE₁ = ATE(
    outcome=:Y₁,
    treatment=(T₁=(case=1, control=0),),
    confounders=[:W₁, :W₂],
    covariates=[:C₁]
)

ATE₂ = ATE(
    outcome=:Y₂,
    treatment=(T₁=(case=1, control=0), T₂=(case="A", control="B")),
    confounders=[:W₁],
)
```
"""
@option struct ATE <: Estimand
    scm::StructuralCausalModel
    outcome::Symbol
    treatment::NamedTuple
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
    - outcome: A symbol identifying the outcome variable of interest
    - treatment: A NamedTuple linking each treatment variable to case/control values
    - confounders: Confounding variables affecting both the outcome and the treatment
    - covariates: Optional extra variables affecting the outcome only

# Examples:
```julia
IATE₁ = IATE(
    outcome=:Y₁,
    treatment=(T₁=(case=1, control=0), T₂=(case="A", control="B")),
    confounders=[:W₁],
)
```
"""
@option struct IATE <: Estimand
    scm::StructuralCausalModel
    outcome::Symbol
    treatment::NamedTuple
end


#####################################################################
###                     Nuisance Estimands                       ###
#####################################################################

"""
# NuisanceEstimands

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
mutable struct NuisanceEstimands
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

Specification of the nuisance estimands to be learnt.

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

CMCompositeEstimand = Union{CM, ATE, IATE}

Base.show(io::IO, Ψ::T) where T <: CMCompositeEstimand = println(io, T)

equations_to_fit(Ψ::CMCompositeEstimand) = (Ψ.scm[outcome(Ψ)], (Ψ.scm[t] for t in treatments(Ψ))...)

variable_not_indataset(role, variable) = string(role, " variable: ", variable, " is not in the dataset.")

function isidentified(Ψ::CMCompositeEstimand, dataset)
    reasons = String[]
    sch = Tables.schema(dataset)
    # Check outcome variable
    outcome(Ψ) ∈ sch.names || push!(
        reasons,
        variable_not_indataset("Outcome", outcome(Ψ))
    )

    # Check Treatment variable
    for treatment in treatments(Ψ)
        treatment ∈ sch.names || push!(
            reasons,
            variable_not_indataset("Treatment", treatment)
        )
    end
    
    # Check adjustment set
    for confounder in Set(vcat(confounders(Ψ)...))
        confounder ∈ sch.names || push!(
            reasons,
            variable_not_indataset("Confounding", confounder)
        )
    end

    return length(reasons) == 0, reasons
end

outcome(Ψ::CMCompositeEstimand) = Ψ.outcome

selectcols(data, cols) = data |> TableOperations.select(cols...) |> Tables.columntable

function confounders(Ψ::CMCompositeEstimand)
    confounders_ = []
    treatments_ = Tuple(treatments(Ψ))
    for treatment in treatments_
        push!(
            confounders_,
            intersect(parents(Ψ.scm, outcome(Ψ)), parents(Ψ.scm, treatment))
        )
    end
    return NamedTuple{treatments_}(confounders_)
end

confounders(dataset, Ψ) = selectcols(dataset, confounders(Ψ))


treatments(Ψ::CMCompositeEstimand) = collect(keys(Ψ.treatment))
treatments(dataset, Ψ) = selectcols(dataset, treatments(Ψ))

outcome(Ψ::Estimand) = Ψ.outcome
outcome(dataset, Ψ) = Tables.getcolumn(dataset, outcome(Ψ))

treatment_and_confounders(Ψ::Estimand) = vcat(confounders(Ψ), treatments(Ψ))

confounders_and_covariates(Ψ::Estimand) = vcat(confounders(Ψ), covariates(Ψ))
confounders_and_covariates(dataset, Ψ) = selectcols(dataset, confounders_and_covariates(Ψ))

"""
Merges together confounders, covariates and floating point representation of treatments.
"""
Qinputs(H, dataset, Ψ::Estimand) = merge(
    columntable(confounders_and_covariates(dataset, Ψ)), 
    MLJBase.transform(H, treatments(dataset, Ψ))
    )


allcolumns(Ψ::Estimand) = vcat(confounders_and_covariates(Ψ), treatments(Ψ), outcome(Ψ))

F_model(::Type{<:AbstractVector{<:MLJBase.Continuous}}) =
    LinearRegressor(fit_intercept=false, offsetcol = :offset)

F_model(::Type{<:AbstractVector{<:Finite}}) =
    LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)

F_model(t::Type{Any}) = throw(ArgumentError("Cannot proceed with Q model with target_scitype $t"))

namedtuples_from_dicts(d) = d
namedtuples_from_dicts(d::Dict) = 
    NamedTuple{Tuple(keys(d))}([namedtuples_from_dicts(val) for val in values(d)])


function param_key(Ψ::Estimand)
    return (
        join(Ψ.confounders, "_"),
        join(keys(Ψ.treatment), "_"),
        string(Ψ.outcome),
        join(Ψ.covariates, "_")
    )
end

"""
    optimize_ordering!(estimands::Vector{<:Estimand})

Reorders the given estimands so that most nuisance estimands fits can be
reused. Given the assumed causal graph:

$causal_graph

and the requirements to estimate both p(T|W) and E[Y|W, T, C].

A natural ordering of the estimands in order to save computations is given by the
following variables ordering: (W, T, Y, C)
"""
optimize_ordering!(estimands::Vector{<:Estimand}) = sort!(estimands, by=param_key)

"""
    optimize_ordering(estimands::Vector{<:Estimand})

See [`optimize_ordering!`](@ref)
"""
optimize_ordering(estimands::Vector{<:Estimand}) = sort(estimands, by=param_key)