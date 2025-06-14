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

Note: This method is for estimands like the ATE or AIE that have case/control treatment settings represented as 
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
    for T in treatments(Ψ)
        treatment_levels = levels(dataset[!, T])
        treatment_settings = Ψ.treatment_values[T]
        check_treatment_settings(treatment_settings, treatment_levels, T)
    end
end

#####################################################################
###                   Conditional Distribution                    ###
#####################################################################
"""
Defines a Conditional Distribution estimand ``(outcome, parents) → P(outcome|parents)``.
"""
@auto_hash_equals struct ConditionalDistribution <: Estimand
    outcome::Symbol
    parents::Tuple{Vararg{Symbol}}
    function ConditionalDistribution(outcome, parents)
        outcome = Symbol(outcome)
        parents = unique_sorted_tuple(parents)
        outcome ∉ parents || throw(ArgumentError(string("The outcome variable of a conditional distribution (here ", outcome, ") cannot be in the parents set.")))
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
###                      JointEstimand                         ###
#####################################################################

@auto_hash_equals struct JointEstimand <: Estimand
    args::Tuple
    JointEstimand(args...) = new(Tuple(args))
end

JointEstimand(;args) = JointEstimand(args...)

function to_dict(Ψ::JointEstimand)
    return Dict(
    :type => string(JointEstimand),
    :args => [to_dict(x) for x in Ψ.args]
)
end

propensity_score_key(Ψ::JointEstimand) = Tuple(unique(Iterators.flatten(propensity_score_key(arg) for arg in Ψ.args)))
outcome_mean_key(Ψ::JointEstimand) = Tuple(unique(outcome_mean_key(arg) for arg in Ψ.args))

n_uniques_nuisance_functions(Ψ::JointEstimand) = length(propensity_score_key(Ψ)) + length(outcome_mean_key(Ψ))

nuisance_functions_iterator(Ψ::JointEstimand) =
    Iterators.flatten(nuisance_functions_iterator(arg) for arg in Ψ.args)

identify(method::AdjustmentMethod, Ψ::JointEstimand, scm) = 
    JointEstimand((identify(method, arg, scm) for arg ∈ Ψ.args)...)

function string_repr(estimand::JointEstimand)
    string(
        "Joint Estimand:\n",
        "--------------\n- ",
        join((string_repr(arg) for arg in estimand.args), "\n- ")
    )
end


#####################################################################
###                       Composed Estimand                       ###
#####################################################################

@auto_hash_equals struct ComposedEstimand <: Estimand
    f::Function
    estimand::JointEstimand
end

ComposedEstimand(;f, estimand) = ComposedEstimand(f, estimand)

ComposedEstimand(f::String, estimand) = ComposedEstimand(eval(Meta.parse(f)), estimand)

function to_dict(Ψ::ComposedEstimand)
    fname = string(parentmodule(Ψ.f), ".", nameof(Ψ.f))
    occursin("#", fname,) && 
        throw(ArgumentError("The function of a ComposedEstimand cannot be anonymous to be converted to a dictionary."))
    return Dict(
        :type => string(ComposedEstimand),
        :f => fname,
        :estimand => to_dict(Ψ.estimand)
    )
end

function string_repr(Ψ::ComposedEstimand)
    firstline = string("Composed Estimand applying function `", Ψ.f , "` to :\n")
    string(
        firstline,
        repeat("-", length(firstline)-2), "\n- ",
        join((string_repr(arg) for arg in Ψ.estimand.args), "\n- ")
    )
end