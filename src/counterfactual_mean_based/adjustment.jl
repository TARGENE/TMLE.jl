#####################################################################
###                    Identification Methods                     ###
#####################################################################

abstract type AdjustmentMethod end

struct BackdoorAdjustment <: AdjustmentMethod
    outcome_extra_covariates::Tuple{Vararg{Symbol}}
    BackdoorAdjustment(outcome_extra_covariates) = new(unique_sorted_tuple(outcome_extra_covariates))
end

BackdoorAdjustment(;outcome_extra_covariates=()) =
    BackdoorAdjustment(outcome_extra_covariates)

function statistical_type_from_causal_type(T) 
    typestring = string(Base.typename(T).wrapper)
    new_typestring = replace(typestring, "TMLE.Causal" => "")
    return eval(Symbol(new_typestring))
end

identify(causal_estimand::T, scm::MetaGraph; method=BackdoorAdjustment()::BackdoorAdjustment) where T<:CausalCMCompositeEstimands =
    identify(method, causal_estimand, scm)


function identify(method::BackdoorAdjustment, causal_estimand::T, scm::MetaGraph) where T<:CausalCMCompositeEstimands
    # Treatment confounders
    treatment_names = keys(causal_estimand.treatment_values)
    treatment_codes = [code_for(scm, treatment) for treatment ∈ treatment_names]
    confounders_codes = scm.graph.badjlist[treatment_codes]
    treatment_confounders = NamedTuple{treatment_names}(
        [[scm.vertex_labels[w] for w in confounders_codes[i]] 
        for i in eachindex(confounders_codes)]
    )

    return statistical_type_from_causal_type(T)(;
        outcome=causal_estimand.outcome,
        treatment_values = causal_estimand.treatment_values,
        treatment_confounders = treatment_confounders,
        outcome_extra_covariates = method.outcome_extra_covariates
    )
end