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
struct ConditionalMean <: Estimand
    scm::StructuralCausalModel
    outcome::Symbol
    treatment::NamedTuple
    function ConditionalMean(scm, outcome, treatment)
        check_parameter_against_scm(scm, outcome, treatment)
        return new(scm, outcome, treatment)
    end
end

const CM = ConditionalMean

CM(;scm, outcome, treatment) = CM(scm, outcome, treatment)
CM(scm; outcome, treatment) = CM(scm, outcome, treatment)

name(::Type{CM}) = "CM"

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
struct AverageTreatmentEffect <: Estimand
    scm::StructuralCausalModel
    outcome::Symbol
    treatment::NamedTuple
    function AverageTreatmentEffect(scm, outcome, treatment)
        check_parameter_against_scm(scm, outcome, treatment)
        return new(scm, outcome, treatment)
    end
end

const ATE = AverageTreatmentEffect

ATE(;scm, outcome, treatment) = ATE(scm, outcome, treatment)
ATE(scm; outcome, treatment) = ATE(scm, outcome, treatment)

name(::Type{ATE}) = "ATE"

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
struct InteractionAverageTreatmentEffect <: Estimand
    scm::StructuralCausalModel
    outcome::Symbol
    treatment::NamedTuple
    function InteractionAverageTreatmentEffect(scm, outcome, treatment)
        check_parameter_against_scm(scm, outcome, treatment)
        return new(scm, outcome, treatment)
    end
end

const IATE = InteractionAverageTreatmentEffect

IATE(;scm, outcome, treatment) = IATE(scm, outcome, treatment)
IATE(scm; outcome, treatment) = IATE(scm, outcome, treatment)

name(::Type{IATE}) = "IATE"

#####################################################################
###                         Methods                               ###
#####################################################################

CMCompositeEstimand = Union{CM, ATE, IATE}

VariableNotAChildInSCMError(variable) = ArgumentError(string("Variable ", variable, " is not associated with a Structural Equation in the SCM."))
TreatmentMustBeInOutcomeParentsError(variable) = ArgumentError(string("Treatment variable ", variable, " must be a parent of the outcome."))

function check_parameter_against_scm(scm::SCM, outcome, treatment)
    eqs = equations(scm)
    haskey(eqs, outcome) || throw(VariableNotAChildInSCMError(outcome))
    for treatment_variable in keys(treatment)
        haskey(eqs, treatment_variable) || throw(VariableNotAChildInSCMError(treatment_variable))
        is_upstream(treatment_variable, outcome, scm) || throw(TreatmentMustBeInOutcomeParentsError(treatment_variable))
    end
end

function Base.show(io::IO, ::MIME"text/plain", Ψ::T) where T <: CMCompositeEstimand 
    param_string = string(
        name(T),
        "\n-----",
        "\nOutcome: ", Ψ.outcome,
        "\nTreatment: ", Ψ.treatment
    )
    println(io, param_string)
end

equations_to_fit(Ψ::CMCompositeEstimand) = (outcome_equation(Ψ), (Ψ.scm[t] for t in treatments(Ψ))...)

outcome_equation(Ψ::CMCompositeEstimand) = Ψ.scm[outcome(Ψ)]

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

confounders(scm, variable, outcome) = 
    intersect(parents(scm, outcome), parents(scm, variable))

function confounders(Ψ::CMCompositeEstimand)
    confounders_ = []
    treatments_ = Tuple(treatments(Ψ))
    for treatment in treatments_
        push!(confounders_, confounders(Ψ.scm, treatment, outcome(Ψ)))
    end
    return NamedTuple{treatments_}(confounders_)
end

function confounders(dataset, Ψ::CMCompositeEstimand)
    confounders_ = []
    treatments_ = Tuple(treatments(Ψ))
    for treatment in treatments_
        push!(
            confounders_,
            selectcols(dataset, confounders(Ψ.scm, treatment, outcome(Ψ)))
        )
    end
    return NamedTuple{treatments_}(confounders_)
end

treatments(Ψ::CMCompositeEstimand) = collect(keys(Ψ.treatment))
treatments(dataset, Ψ::CMCompositeEstimand) = selectcols(dataset, treatments(Ψ))

outcome(Ψ::CMCompositeEstimand) = Ψ.outcome
outcome(dataset, Ψ::CMCompositeEstimand) = Tables.getcolumn(dataset, outcome(Ψ))

F_model(::Type{<:AbstractVector{<:MLJBase.Continuous}}) =
    LinearRegressor(fit_intercept=false, offsetcol = :offset)

F_model(::Type{<:AbstractVector{<:Finite}}) =
    LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)

F_model(t::Type{Any}) = throw(ArgumentError("Cannot proceed with Q model with target_scitype $t"))

function param_key(Ψ::CMCompositeEstimand)
    return (
        join(values(confounders(Ψ))..., "_"),
        join(treatments(Ψ), "_"),
        string(outcome(Ψ)),
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