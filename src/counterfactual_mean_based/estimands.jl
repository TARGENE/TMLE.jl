#####################################################################
###                      Counterfactual Mean                      ###
#####################################################################

"""
# Counterfactual Mean / CM

## Definition

``CM(Y, T=t) = E[Y|do(T=t)]``

## Constructors

- CM(scm::SCM; outcome::Symbol, treatment::NamedTuple)
- CM(
    outcome::Symbol, 
    treatment::NamedTuple, 
    confounders::Union{Symbol, AbstractVector{Symbol}}, 
    covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
    outcome_model = with_encoder(LinearRegressor()),
    treatment_model = LinearBinaryClassifier()
)
## Example

```julia
Ψ = CM(scm, outcome=:Y, treatment=(T=1,))

```
"""
struct CounterfactualMean <: Estimand
    scm::StructuralCausalModel
    outcome::Symbol
    treatment::NamedTuple
    function CounterfactualMean(scm, outcome, treatment)
        check_parameter_against_scm(scm, outcome, treatment)
        return new(scm, outcome, treatment)
    end
end

const CM = CounterfactualMean

indicator_fns(Ψ::CM) = Dict(values(Ψ.treatment) => 1.)


#####################################################################
###                  Average Treatment Effect                     ###
#####################################################################
"""
# Average Treatment Effect / ATE

## Definition

``ATE(Y, T, case, control) = E[Y|do(T=case)] - E[Y|do(T=control)``

## Constructors

- ATE(scm::SCM; outcome::Symbol, treatment::NamedTuple)
- ATE(
    outcome::Symbol, 
    treatment::NamedTuple, 
    confounders::Union{Symbol, AbstractVector{Symbol}}, 
    covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
    outcome_model = with_encoder(LinearRegressor()),
    treatment_model = LinearBinaryClassifier()
)

## Example

```julia
Ψ = ATE(scm, outcome=:Y, treatment=(T=(case=1,control=0),)
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

function indicator_fns(Ψ::ATE)
    case = []
    control = []
    for treatment in Ψ.treatment
        push!(case, treatment.case)
        push!(control, treatment.control)
    end
    return Dict(Tuple(case) => 1., Tuple(control) => -1.)
end

#####################################################################
###            Interaction Average Treatment Effect               ###
#####################################################################

"""
# Interaction Average Treatment Effect / IATE

## Definition

For two treatments with case/control settings (1, 0):

``IATE = E[Y|do(T₁=1, T₂=1)] - E[Y|do(T₁=1, T₂=0)] - E[Y|do(T₁=0, T₂=1)] + E[Y|do(T₁=0, T₂=0)]``

## Constructors

- IATE(scm::SCM; outcome::Symbol, treatment::NamedTuple)
- IATE(
    outcome::Symbol, 
    treatment::NamedTuple, 
    confounders::Union{Symbol, AbstractVector{Symbol}}, 
    covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
    outcome_model = with_encoder(LinearRegressor()),
    treatment_model = LinearBinaryClassifier()
)

## Example

```julia
Ψ = IATE(scm, outcome=:Y, treatment=(T₁=(case=1,control=0), T₂=(case=1,control=0))
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

ncases(value, Ψ::IATE) = sum(value[i] == Ψ.treatment[i].case for i in eachindex(value))

function indicator_fns(Ψ::IATE)
    N = length(treatments(Ψ))
    key_vals = Pair[]
    for cf in Iterators.product((values(Ψ.treatment[T]) for T in treatments(Ψ))...)
        push!(key_vals, cf => float((-1)^(N - ncases(cf, Ψ))))
    end
    return Dict(key_vals...)
end

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

        $(typename)(scm::SCM; outcome::Symbol, treatment::NamedTuple) = $(typename)(scm, outcome, treatment)
        
        function $(typename)(;
            outcome::Symbol, 
            treatment::NamedTuple, 
            confounders::Union{Symbol, AbstractVector{Symbol}}, 
            covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing
            )
            scm = StaticConfoundedModel(
                [outcome], 
                collect(keys(treatment)), 
                confounders;
                covariates=covariates
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

function get_relevant_factors(Ψ::CMCompositeEstimand; adjustment_method=BackdoorAdjustment())
    outcome_model = ExpectedValue(Ψ.scm, outcome(Ψ), outcome_parents(adjustment_method, Ψ))
    treatment_factors = Tuple(ConditionalDistribution(Ψ.scm, treatment, parents(Ψ.scm, treatment)) for treatment in treatments(Ψ))
    return CMRelevantFactors(Ψ.scm, outcome_model, treatment_factors)
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

get_outcome_model(Ψ::CMCompositeEstimand) = outcome_equation(Ψ).mach

get_outcome_datas(Ψ::CMCompositeEstimand) = get_outcome_model(Ψ).data

outcome(Ψ::CMCompositeEstimand) = Ψ.outcome
outcome(dataset, Ψ::CMCompositeEstimand) = Tables.getcolumn(dataset, outcome(Ψ))

function estimand_key(Ψ::CMCompositeEstimand)
    return (
        join(treatments(Ψ), "_"),
        string(outcome(Ψ)),
    )
end
