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

treatments(Ψ::Estimand) = collect(keys(Ψ.treatment_values))

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
        treatment_settings = getproperty(Ψ.treatment_values, treatment_name)
        check_treatment_settings(treatment_settings, treatment_levels, treatment_name)
    end
end

"""
Function used to sort estimands for optimal estimation ordering.
"""
key(estimand::Estimand) = estimand

#####################################################################
###                   Conditional Distribution                    ###
#####################################################################
"""
Defines a Conditional Distribution estimand ``(outcome, parents) → P(outcome|parents)``.
"""
struct ConditionalDistribution <: Estimand
    outcome::Symbol
    parents::Tuple{Vararg{Symbol}}
    function ConditionalDistribution(outcome, parents)
        outcome = Symbol(outcome)
        parents = unique_sorted_tuple(parents)
        outcome ∉ parents || throw(SelfReferringEquationError(outcome))
        # Maybe check variables are in the SCM?
        return new(outcome, parents)
    end
end

string_repr(estimand::ConditionalDistribution) = 
    string("P₀(", estimand.outcome, " | ", join(estimand.parents, ", "), ")")

variables(estimand::ConditionalDistribution) = (estimand.outcome, estimand.parents...)

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
###                      ComposedEstimand                         ###
#####################################################################

struct ComposedEstimand <: Estimand
    f::Function
    args::Tuple
end