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

identify(estimand, scm; method=BackdoorAdjustment()) =
    identify(method, estimand, scm)


to_dict(adjustment::BackdoorAdjustment) = Dict(
    :type => "BackdoorAdjustment",
    :outcome_extra_covariates => collect(adjustment.outcome_extra_covariates)
)