
abstract type AdjustmentMethod end

struct BackdoorAdjustment <: AdjustmentMethod
    outcome_extra::Vector{Symbol}
end

"""
    BackdoorAdjustment(;outcome_extra=[])

The adjustment set for each treatment variable is simply the set of direct parents in the 
associated structural model.

`outcome_extra` are optional additional variables that can be used to fit the outcome model 
in order to improve inference.
"""
BackdoorAdjustment(;outcome_extra=[]) = BackdoorAdjustment(outcome_extra)

"""
    outcome_input_variables(treatments, confounding_variables)

This is defined as the set of variable corresponding to:

- Treatment variables
- Confounding variables
- Extra covariates
"""
outcome_input_variables(treatments, confounding_variables, extra_covariates) = unique(vcat(
    treatments, 
    confounding_variables, 
    extra_covariates
))

function get_models_input_variables(adjustment_method::BackdoorAdjustment, Ψ::CMCompositeEstimand)
    models_inputs = []
    for treatment in treatments(Ψ)
        push!(models_inputs, parents(Ψ.scm, treatment))
    end
    unique_confounders = unique(vcat(models_inputs...))
    push!(
        models_inputs, 
        outcome_input_variables(treatments(Ψ), unique_confounders, adjustment_method.outcome_extra)
    )
    variables = Tuple(vcat(treatments(Ψ), outcome(Ψ)))
    return NamedTuple{variables}(models_inputs)
end
