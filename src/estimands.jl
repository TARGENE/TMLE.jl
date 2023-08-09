#####################################################################
###                    Abstract Estimand                          ###
#####################################################################
"""
A Estimand is a functional on distribution space Ψ: ℳ → ℜ. 
"""
abstract type Estimand end

#####################################################################
###                      Conditional Mean                         ###
#####################################################################

"""
# Conditional Mean / CM

## Definition

``CM(Y, T=t) = E[Y|do(T=t)]``

## Constructors

- CM(;scm::SCM, outcome, treatment)
- CM(scm::SCM; outcome, treatment)

where:

- scm: is a `StructuralCausalModel` (see [`SCM`](@ref))
- outcome: is a `Symbol`
- treatment: is a `NamedTuple`

## Example

Ψ = CM(scm, outcome=:Y, treatment=(T=1,))
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

#####################################################################
###                  Average Treatment Effect                     ###
#####################################################################
"""
# Average Treatment Effect / ATE

## Definition

``ATE(Y, T, case, control) = E[Y|do(T=case)] - E[Y|do(T=control)``

## Constructors

- ATE(;scm::SCM, outcome, treatment)
- ATE(scm::SCM; outcome, treatment)

where:

- scm: is a `StructuralCausalModel` (see [`SCM`](@ref))
- outcome: is a `Symbol`
- treatment: is a `NamedTuple`

## Example

Ψ = ATE(scm, outcome=:Y, treatment=(T=(case=1,control=0),)
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

#####################################################################
###            Interaction Average Treatment Effect               ###
#####################################################################

"""
# Interaction Average Treatment Effect / IATE

## Definition

For two treatments with settings (1, 0):

``IATE = E[Y|do(T₁=1, T₂=1)] - E[Y|do(T₁=1, T₂=0)] - E[Y|do(T₁=0, T₂=1)] + E[Y|do(T₁=0, T₂=0)]``

## Constructors

- IATE(;scm::SCM, outcome, treatment)
- IATE(scm::SCM; outcome, treatment)

where:

- scm: is a `StructuralCausalModel` (see [`SCM`](@ref))
- outcome: is a `Symbol`
- treatment: is a `NamedTuple`

## Example

Ψ = IATE(scm, outcome=:Y, treatment=(T₁=(case=1,control=0), T₂=(case=1,control=0))
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

#####################################################################
###                         Methods                               ###
#####################################################################

AVAILABLE_ESTIMANDS = (CM, ATE, IATE)
CMCompositeTypenames = [:CM, :ATE, :IATE]
CMCompositeEstimand = Union{(eval(x) for x in CMCompositeTypenames)...}

# Define constructors/name for CMCompositeEstimand types
for typename in CMCompositeTypenames
    ex = quote
        name(::Type{$(typename)}) = string($(typename))

        $(typename)(;scm::SCM, outcome::Symbol, treatment::NamedTuple) = $(typename)(scm, outcome, treatment)
        $(typename)(scm::SCM; outcome::Symbol, treatment::NamedTuple) = $(typename)(scm, outcome, treatment)
        
        function $(typename)(
            outcome::Symbol, treatment::NamedTuple, confounders::Union{Symbol, AbstractVector{Symbol}}; 
            covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
            outcome_model = with_encoder(LinearRegressor()),
            treatment_model = LinearBinaryClassifier())
            scm = StaticConfoundedModel(outcome, collect(keys(treatment)), confounders;
                covariates=covariates,
                outcome_model=outcome_model,
                treatment_model=treatment_model
                )
            return $(typename)(scm; outcome=outcome, treatment=treatment)
        end
    end

    eval(ex)
end

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

outcome_equation(Ψ::CMCompositeEstimand) = Ψ.scm[outcome(Ψ)]

treatments(Ψ::CMCompositeEstimand) = collect(keys(Ψ.treatment))
treatments(dataset, Ψ::CMCompositeEstimand) = selectcols(dataset, treatments(Ψ))

outcome(Ψ::CMCompositeEstimand) = Ψ.outcome
outcome(dataset, Ψ::CMCompositeEstimand) = Tables.getcolumn(dataset, outcome(Ψ))

confounders(dataset, Ψ) = (;(T => selectcols(dataset, keys(Ψ.scm[T].mach.data[1])) for T in treatments(Ψ))...)

F_model(::Type{<:AbstractVector{<:MLJBase.Continuous}}) =
    LinearRegressor(fit_intercept=false, offsetcol = :offset)

F_model(::Type{<:AbstractVector{<:Finite}}) =
    LinearBinaryClassifier(fit_intercept=false, offsetcol = :offset)

F_model(t::Type{Any}) = throw(ArgumentError("Cannot proceed with Q model with target_scitype $t"))

function param_key(Ψ::CMCompositeEstimand)
    return (
        join(treatments(Ψ), "_"),
        string(outcome(Ψ)),
    )
end

"""
    optimize_ordering!(estimands::Vector{<:Estimand})

Optimizes the order of the `estimands` to maximize reuse of 
fitted equations in the associated SCM.
"""
optimize_ordering!(estimands::Vector{<:Estimand}) = sort!(estimands, by=param_key)

"""
    optimize_ordering(estimands::Vector{<:Estimand})

See [`optimize_ordering!`](@ref)
"""
optimize_ordering(estimands::Vector{<:Estimand}) = sort(estimands, by=param_key)