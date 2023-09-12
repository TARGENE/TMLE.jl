#####################################################################
###                    Abstract Estimand                          ###
#####################################################################
"""
A Estimand is a functional on distribution space Ψ: ℳ → ℜ. 
"""
abstract type Estimand end

string_repr(estimand::Estimand) = estimand

Base.show(io::IO, ::MIME"text/plain", estimand::Estimand) =
    println(io, string_repr(estimand))

treatments(Ψ::Estimand) = collect(keys(Ψ.treatment))

AbsentLevelError(treatment_name, key, val, levels) = ArgumentError(string(
    "The treatment variable ", treatment_name, "'s, '", key, "' level: '", val,
    "' in Ψ does not match any level in the dataset: ", levels))

AbsentLevelError(treatment_name, val, levels) = ArgumentError(string(
    "The treatment variable ", treatment_name, "'s, level: '", val,
    "' in Ψ does not match any level in the dataset: ", levels))

"""
    check_treatment_settings(settings::NamedTuple, levels, treatment_name)

Checks the case/control values defining the treatment contrast are present in the dataset levels. 

Note: This method is for estimands like the ATE or IATE that have case/control treatment settings represented as 
`NamedTuple`.
"""
function check_treatment_settings(settings::NamedTuple, levels, treatment_name)
    for (key, val) in zip(keys(settings), settings) 
        any(val .== levels) || 
            throw(AbsentLevelError(treatment_name, key, val, levels))
    end
end

"""
    check_treatment_settings(setting, levels, treatment_name)

Checks the value defining the treatment setting is present in the dataset levels. 

Note: This is for estimands like the CM that do not have case/control treatment settings 
and are represented as simple values.
"""
function check_treatment_settings(setting, levels, treatment_name)
    any(setting .== levels) || 
            throw(
                AbsentLevelError(treatment_name, setting, levels))
end

"""
    check_treatment_levels(Ψ::Estimand, dataset)

Makes sure the defined treatment levels are present in the dataset.
"""
function check_treatment_levels(Ψ::Estimand, dataset)
    for treatment_name in treatments(Ψ)
        treatment_levels = levels(Tables.getcolumn(dataset, treatment_name))
        treatment_settings = getproperty(Ψ.treatment, treatment_name)
        check_treatment_settings(treatment_settings, treatment_levels, treatment_name)
    end
end

"""
Function used to sort estimands for optimal estimation ordering.
"""
function estimand_key end

"""
    optimize_ordering!(estimands::Vector{<:Estimand})

Optimizes the order of the `estimands` to maximize reuse of 
fitted equations in the associated SCM.
"""
optimize_ordering!(estimands::Vector{<:Estimand}) = sort!(estimands, by=estimand_key)

"""
    optimize_ordering(estimands::Vector{<:Estimand})

See [`optimize_ordering!`](@ref)
"""
optimize_ordering(estimands::Vector{<:Estimand}) = sort(estimands, by=estimand_key)

#####################################################################
###                   Conditional Distribution                    ###
#####################################################################
"""
Defines a Conditional Distribution estimand ``(outcome, parents) → P(outcome|parents)``.
"""
struct ConditionalDistribution <: Estimand
    scm::SCM
    outcome::Symbol
    parents::Set{Symbol}
    function ConditionalDistribution(scm, outcome, parents)
        outcome = Symbol(outcome)
        parents = Set(Symbol(x) for x in parents)
        outcome ∉ parents || throw(SelfReferringEquationError(outcome))
        # Maybe check variables are in the SCM?
        return new(scm, outcome, parents)
    end
end

string_repr(estimand::ConditionalDistribution) = 
    string("P₀(", estimand.outcome, " | ", join(estimand.parents, ", "), ")")

featurenames(estimand::ConditionalDistribution) = sort(collect(estimand.parents))

variables(estimand::ConditionalDistribution) = union(Set([estimand.outcome]), estimand.parents)

estimand_key(cd::ConditionalDistribution) = (cd.outcome, cd.parents)

#####################################################################
###                        ExpectedValue                          ###
#####################################################################

"""
Defines an Expected Value estimand ``parents → E[Outcome|parents]``.
At the moment there is no distinction between an Expected Value and 
a Conditional Distribution because they are estimated in the same way.
"""
const ExpectedValue = ConditionalDistribution

#####################################################################
###                       CMRelevantFactors                       ###
#####################################################################

"""
Defines relevant factors that need to be estimated in order to estimate any
Counterfactual Mean composite estimand (see `CMCompositeEstimand`).
"""
struct CMRelevantFactors <: Estimand
    scm::SCM
    outcome_mean::ConditionalDistribution
    propensity_score::Tuple{Vararg{ConditionalDistribution}}
end

CMRelevantFactors(scm, outcome_mean, propensity_score::ConditionalDistribution) = 
    CMRelevantFactors(scm, outcome_mean, (propensity_score,))

CMRelevantFactors(scm; outcome_mean, propensity_score) = CMRelevantFactors(scm, outcome_mean, propensity_score)

string_repr(estimand::CMRelevantFactors) = 
    string("Composite Factor: \n",
           "----------------\n- ",
        string_repr(estimand.outcome_mean),"\n- ", 
        join((string_repr(f) for f in estimand.propensity_score), "\n- "))

variables(estimand::CMRelevantFactors) = 
    union(variables(estimand.outcome_mean), (variables(est) for est in estimand.propensity_score)...)

"""
    estimand_key(estimand::CMRelevantFactors)

The key combines the outcome_mean's key anc the propensity scores' keys. 
Because the order of treatment does not matter, we order them by treatment.
"""
estimand_key(estimand::CMRelevantFactors) = (
    estimand_key(estimand.outcome_mean),
    sort((estimand_key(ps) for ps in estimand.propensity_score), by=x->x[1])...
    )