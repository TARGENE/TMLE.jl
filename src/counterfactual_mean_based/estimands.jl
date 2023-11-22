#####################################################################
###                       CMRelevantFactors                       ###
#####################################################################

"""
Defines relevant factors that need to be estimated in order to estimate any
Counterfactual Mean composite estimand (see `StatisticalCMCompositeEstimand`).
"""
struct CMRelevantFactors <: Estimand
    outcome_mean::ConditionalDistribution
    propensity_score::Tuple{Vararg{ConditionalDistribution}}
end

CMRelevantFactors(outcome_mean, propensity_score::ConditionalDistribution) = 
    CMRelevantFactors(outcome_mean, (propensity_score,))

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

const ESTIMANDS_DOCS = Dict(
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

            function $(causal_estimand)(outcome, treatment_values)
                outcome = Symbol(outcome)
                treatment_values = get_treatment_specs(treatment_values)
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
                treatment_values = get_treatment_specs(treatment_values)
                treatment_variables = Tuple(keys(treatment_values))
                treatment_confounders = NamedTuple{treatment_variables}([unique_sorted_tuple(treatment_confounders[T]) for T ∈ treatment_variables])
                outcome_extra_covariates = unique_sorted_tuple(outcome_extra_covariates)
                return new(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
            end
        end

        # Constructors
        $(causal_estimand)(;outcome, treatment_values) =  $(causal_estimand)(outcome, treatment_values)

        $(statistical_estimand)(;outcome, treatment_values, treatment_confounders, outcome_extra_covariates) =
            $(statistical_estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
        
        # Short Name Constructors
        $(estimand)(outcome, treatment_values) = $(causal_estimand)(outcome, treatment_values)

        $(estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates) = 
            $(statistical_estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)

        $(estimand)(outcome, treatment_values, treatment_confounders::Nothing, outcome_extra_covariates) = 
            $(causal_estimand)(outcome, treatment_values)

        $(estimand)(;outcome, treatment_values, treatment_confounders=nothing, outcome_extra_covariates=()) =
            $(estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)

    end
    eval(ex)
end

CausalCMCompositeEstimands = Union{(eval(Symbol(:Causal, x)) for x in keys(ESTIMANDS_DOCS))...}

StatisticalCMCompositeEstimand = Union{(eval(Symbol(:Statistical, x)) for x in keys(ESTIMANDS_DOCS))...}

const AVAILABLE_ESTIMANDS = [x[1] for x ∈ ESTIMANDS_DOCS]

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

function get_relevant_factors(Ψ::StatisticalCMCompositeEstimand)
    outcome_model = ExpectedValue(Ψ.outcome, Tuple(union(Ψ.outcome_extra_covariates, keys(Ψ.treatment_confounders), (Ψ.treatment_confounders)...)))
    treatment_factors = Tuple(ConditionalDistribution(T, Ψ.treatment_confounders[T]) for T in treatments(Ψ))
    return CMRelevantFactors(outcome_model, treatment_factors)
end

function Base.show(io::IO, ::MIME"text/plain", Ψ::T) where T <: StatisticalCMCompositeEstimand 
    param_string = string(
        Base.typename(T).wrapper,
        "\n-----",
        "\nOutcome: ", Ψ.outcome,
        "\nTreatment: ", Ψ.treatment_values
    )
    println(io, param_string)
end

function treatment_specs_to_dict(treatment_values::NamedTuple{T, <:Tuple{Vararg{NamedTuple}}}) where T
    Dict(key => Dict(pairs(vals)) for (key, vals) in pairs(treatment_values))
end

treatment_specs_to_dict(treatment_values::NamedTuple) = Dict(pairs(treatment_values))

treatment_values(d::AbstractDict) = (;d...)
treatment_values(d) = d

get_treatment_specs(treatment_specs::NamedTuple{names, }) where names = 
    NamedTuple{sort(names)}(treatment_specs)

function get_treatment_specs(treatment_specs::NamedTuple{names, <:Tuple{Vararg{NamedTuple}}}) where names
    case_control = ((case=v[:case], control=v[:control]) for v in values(treatment_specs))
    treatment_specs = (;zip(keys(treatment_specs), case_control)...)
    return NamedTuple{sort(names)}(treatment_specs)
end
    
get_treatment_specs(treatment_specs::AbstractDict) = 
    get_treatment_specs((;(key => treatment_values(val) for (key, val) in treatment_specs)...))

constructorname(T; prefix="TMLE.Causal") = replace(string(T), prefix => "")

treatment_confounders_to_dict(treatment_confounders::NamedTuple) = 
    Dict(key => collect(vals) for (key, vals) in pairs(treatment_confounders))

"""
    to_dict(Ψ::T) where T <: CausalCMCompositeEstimands

Converts Ψ to a dictionary that can be serialized.
"""
function to_dict(Ψ::T) where T <: CausalCMCompositeEstimands
    return Dict(
        :type => constructorname(T; prefix="TMLE.Causal"),
        :outcome => Ψ.outcome,
        :treatment_values => treatment_specs_to_dict(Ψ.treatment_values)
        )
end

"""
    to_dict(Ψ::T) where T <: StatisticalCMCompositeEstimand

Converts Ψ to a dictionary that can be serialized.
"""
function to_dict(Ψ::T) where T <: StatisticalCMCompositeEstimand
    return Dict(
        :type => constructorname(T; prefix="TMLE.Statistical"),
        :outcome => Ψ.outcome,
        :treatment_values => treatment_specs_to_dict(Ψ.treatment_values),
        :treatment_confounders => treatment_confounders_to_dict(Ψ.treatment_confounders),
        :outcome_extra_covariates => collect(Ψ.outcome_extra_covariates)
        )
end