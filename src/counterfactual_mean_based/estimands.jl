#####################################################################
###                       CMRelevantFactors                       ###
#####################################################################

"""
Defines relevant factors that need to be estimated in order to estimate any
Counterfactual Mean composite estimand (see `CMCompositeEstimand`).
"""
struct CMRelevantFactors <: Estimand
    outcome_mean::ConditionalDistribution
    propensity_score::Tuple{Vararg{ConditionalDistribution}}
end

CMRelevantFactors(outcome_mean, propensity_score::ConditionalDistribution) = 
    CMRelevantFactors(outcome_mean, (propensity_score,))

CMRelevantFactors(outcome_mean, propensity_score) = 
    CMRelevantFactors(outcome_mean, propensity_score)

CMRelevantFactors(;outcome_mean, propensity_score) = 
    CMRelevantFactors(outcome_mean, propensity_score)

string_repr(estimand::CMRelevantFactors) = 
    string("Composite Factor: \n",
           "----------------\n- ",
        string_repr(estimand.outcome_mean),"\n- ", 
        join((string_repr(f) for f in estimand.propensity_score), "\n- "))

variables(estimand::CMRelevantFactors) = 
    Tuple(union(variables(estimand.outcome_mean), (variables(est) for est in estimand.propensity_score)...))

#####################################################################
###                         Functionals                           ###
#####################################################################

ESTIMANDS_DOCS = Dict(
    :CM => (formula="``CM(Y, T=t) = E[Y|do(T=t)]``",),
    :ATE => (formula="``ATE(Y, T, case, control) = E[Y|do(T=case)] - E[Y|do(T=control)``",),
    :IATE => (formula="``IATE = E[Y|do(T₁=1, T₂=1)] - E[Y|do(T₁=1, T₂=0)] - E[Y|do(T₁=0, T₂=1)] + E[Y|do(T₁=0, T₂=0)]``",)
)

for (estimand, (formula,)) ∈ ESTIMANDS_DOCS
    causal_estimand = Symbol(:Causal, estimand)
    statistical_estimand = Symbol(:Statistical, estimand)
    ex = quote
        # Causal Estimand
        struct $(causal_estimand) <: Estimand
            outcome::Symbol
            treatment_values::NamedTuple

            function $(statistical_estimand)(outcome, treatment_values)
                outcome = Symbol(outcome)
                treatment_variables = Tuple(keys(treatment_values))
                return new(outcome, treatment_values)
            end
        end
        # Statistical Estimand
        struct $(statistical_estimand) <: Estimand
            outcome::Symbol
            treatment_values::NamedTuple
            treatment_confounders::NamedTuple
            outcome_extra_covariates::Tuple{Vararg{Symbol}}

            function $(statistical_estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
                outcome = Symbol(outcome)
                treatment_variables = Tuple(keys(treatment_values))
                treatment_confounders = NamedTuple{treatment_variables}([unique_sorted_tuple(treatment_confounders[T]) for T ∈ treatment_variables])
                outcome_extra_covariates = unique_sorted_tuple(outcome_extra_covariates)
                return new(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
            end
        end

        # Constructors
        $(estimand)(outcome, treatment_values) = $(causal_estimand)(outcome, treatment_values)

        $(estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates) = 
            $(statistical_estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)

        $(estimand)(outcome, treatment_values, treatment_confounders::Nothing, outcome_extra_covariates) = 
            $(statistical_estimand)(outcome, treatment_values)

        $(estimand)(;outcome, treatment_values, treatment_confounders, outcome_extra_covariates=()) =
            $(estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)

    end
    eval(ex)
end

CMCompositeEstimand = Union{(eval(Symbol(:Statistical, x)) for x in keys(ESTIMANDS_DOCS))...}

indicator_fns(Ψ::StatisticalCM) = Dict(values(Ψ.treatment_values) => 1.)

function indicator_fns(Ψ::StatisticalATE)
    case = []
    control = []
    for treatment in Ψ.treatment_values
        push!(case, treatment.case)
        push!(control, treatment.control)
    end
    return Dict(Tuple(case) => 1., Tuple(control) => -1.)
end

ncases(value, Ψ::StatisticalIATE) = sum(value[i] == Ψ.treatment_values[i].case for i in eachindex(value))

function indicator_fns(Ψ::StatisticalIATE)
    N = length(treatments(Ψ))
    key_vals = Pair[]
    for cf in Iterators.product((values(Ψ.treatment_values[T]) for T in treatments(Ψ))...)
        push!(key_vals, cf => float((-1)^(N - ncases(cf, Ψ))))
    end
    return Dict(key_vals...)
end

function get_relevant_factors(Ψ::CMCompositeEstimand)
    outcome_model = ExpectedValue(Ψ.outcome, Tuple(union(Ψ.outcome_extra_covariates, keys(Ψ.treatment_confounders), (Ψ.treatment_confounders)...)))
    treatment_factors = Tuple(ConditionalDistribution(T, Ψ.treatment_confounders[T]) for T in treatments(Ψ))
    return CMRelevantFactors(outcome_model, treatment_factors)
end

function Base.show(io::IO, ::MIME"text/plain", Ψ::T) where T <: CMCompositeEstimand 
    param_string = string(
        Base.typename(T).wrapper,
        "\n-----",
        "\nOutcome: ", Ψ.outcome,
        "\nTreatment: ", Ψ.treatment_values
    )
    println(io, param_string)
end