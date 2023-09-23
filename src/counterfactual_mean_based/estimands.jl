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
# Define constructors/name for CMCompositeEstimand types

for (typename, (formula,)) ∈ ESTIMANDS_DOCS
    ex = quote
        # """
        # # $(typename)

        # ## Definition

        # For two treatments with case/control settings (1, 0):

        # $(formula)

        # ## Constructors

        # - $(typename)(outcome, treatment, confounders; extra_outcome_covariates=Set{Symbol}())
        # - $(typename)(;
        #     outcome, 
        #     treatment, 
        #     confounders, 
        #     extra_outcome_covariates 
        # )

        # ## Example

        # ```julia
        # Ψ = $(typename)(outcome=:Y, treatment=(T₁=(case=1, control=0), T₂=(case=1,control=0), confounders=[:W₁, :W₂])
        # ```
        # """
        struct $(typename) <: Estimand
            outcome::Symbol
            treatment_values::NamedTuple
            treatment_confounders::NamedTuple
            outcome_extra_covariates::Tuple{Vararg{Symbol}}

            function $(typename)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
                outcome = Symbol(outcome)
                treatment_variables = Tuple(keys(treatment_values))
                treatment_confounders = NamedTuple{treatment_variables}([unique_sorted_tuple(treatment_confounders[T]) for T ∈ treatment_variables])
                outcome_extra_covariates = unique_sorted_tuple(outcome_extra_covariates)
                return new(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
            end
        end

        $(typename)(outcome, treatment_values, treatment_confounders; outcome_extra_covariates=()) = 
            $(typename)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
    
        $(typename)(;outcome, treatment_values, treatment_confounders, outcome_extra_covariates=()) = 
            $(typename)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)

        name(::Type{$(typename)}) = string($(typename))
    end
    eval(ex)
end

CMCompositeEstimand = Union{(eval(x) for x in keys(ESTIMANDS_DOCS))...}

indicator_fns(Ψ::CM) = Dict(values(Ψ.treatment_values) => 1.)

function indicator_fns(Ψ::ATE)
    case = []
    control = []
    for treatment in Ψ.treatment_values
        push!(case, treatment.case)
        push!(control, treatment.control)
    end
    return Dict(Tuple(case) => 1., Tuple(control) => -1.)
end

ncases(value, Ψ::IATE) = sum(value[i] == Ψ.treatment_values[i].case for i in eachindex(value))

function indicator_fns(Ψ::IATE)
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
        name(T),
        "\n-----",
        "\nOutcome: ", Ψ.outcome,
        "\nTreatment: ", Ψ.treatment_values
    )
    println(io, param_string)
end