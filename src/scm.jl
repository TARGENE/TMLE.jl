#####################################################################
###                  Structural Equation                          ###
#####################################################################

function SCM(equations...)
    scm =  MetaGraph(
        SimpleDiGraph();
        label_type=Symbol,
        vertex_data_type=Nothing,
        edge_data_type=Nothing
    )
    for (outcome, parents) in equations
        add_equation!(scm, outcome, parents)
    end
    return scm
end

function add_equation!(scm::MetaGraph, outcome, parents)
    outcome_symbol = Symbol(outcome)
    add_vertex!(scm, outcome_symbol)
    for parent in parents
        parent_symbol = Symbol(parent)
        add_vertex!(scm, parent_symbol)
        add_edge!(scm, parent_symbol, outcome_symbol)
    end
end


get_outcome_extra_covariates(outcome_parents, treatment_and_confounders, vertex_labels) = 
    [vertex_labels[p] for p ∈ outcome_parents if p ∉ treatment_and_confounders]

#####################################################################
###                    Identification Methods                     ###
#####################################################################

abstract type AdjustmentMethod end

struct BackdoorAdjustment <: AdjustmentMethod
    outcome_extra::Bool
end

function identify(method::BackdoorAdjustment, estimand::T, scm::MetaGraph) where T<:Union{Cau}
    # Treatment confounders
    treatment_names = keys(estimand.treatment)
    treatment_codes = [code_for(scm, treatment) for treatment ∈ treatment_names]
    confounders_codes = scm.graph.badjlist[treatment_codes]
    treatment_confounders = NamedTuple{treatment_names}(
        [[scm.vertex_labels[w] for w in confounders_codes[i]] 
        for i in eachindex(confounders_codes)]
    )
    # Extra covariates
    outcome_extra_covariates = nothing
    if method.outcome_extra_covariates
        outcome_parents_codes = scm.graph.badjlist[code_for(scm, estimand.outcome)]
        treatment_and_confounders_codes = Set(vcat(treatment_codes, confounders_codes...))
        outcome_extra_covariates = get_outcome_extra_covariates(
            outcome_parents_codes, 
            treatment_and_confounders_codes, 
            scm.vertex_labels
        )
    end

    return T(
        outcome=estimand.outcome,
        treatment_values = estimand.treatment_values,
        treatment_confounders = treatment_confounders,
        outcome_extra_covariates = outcome_extra_covariates
    )
end

#####################################################################
###                      Causal Estimand                          ###
#####################################################################
struct CausalATE
    outcome::Symbol
    treatment_values::NamedTuple
end