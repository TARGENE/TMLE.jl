#####################################################################
###                  Structural Equation                          ###
#####################################################################

"""
A SCM is simply a MetaGraph over a Directed Acyclic Graph with additional methods.
"""
const SCM = MetaGraph

function SCM(equations...)
    scm =  MetaGraph(
        SimpleDiGraph();
        label_type=Symbol,
        vertex_data_type=Nothing,
        edge_data_type=Nothing
    )
    add_equations!(scm, equations...)
    return scm
end

function add_equations!(scm::SCM, equations...)
    for outcome_parents_pair in equations
        add_equation!(scm, outcome_parents_pair)
    end
end

function add_equation!(scm::SCM, outcome_parents_pair)
    outcome, parents = outcome_parents_pair
    outcome_symbol = Symbol(outcome)
    add_vertex!(scm, outcome_symbol)
    for parent in parents
        parent_symbol = Symbol(parent)
        add_vertex!(scm, parent_symbol)
        add_edge!(scm, parent_symbol, outcome_symbol)
    end
end

function parents(scm, label)
    code, _ = scm.vertex_properties[label]
    return [scm.vertex_labels[parent_code] for parent_code in scm.graph.badjlist[code]]
end

"""
A plate Structural Causal Model where:

- For all outcomes: oᵢ = fᵢ(treatments, confounders, outcome_extra_covariates)
- For all treatments: tⱼ = fⱼ(confounders)

#  Example

StaticSCM([:Y], [:T₁, :T₂], [:W₁, :W₂, :W₃]; outcome_extra_covariates=[:C])
"""
function StaticSCM(outcomes, treatments, confounders; outcome_extra_covariates=())
    outcome_equations = (outcome => unique(vcat(treatments, confounders, outcome_extra_covariates)) for outcome in outcomes)
    treatment_equations = (treatment => unique(confounders) for treatment in treatments)
    return SCM(outcome_equations..., treatment_equations...)
end

StaticSCM(;outcomes, treatments, confounders, outcome_extra_covariates=()) = 
    StaticSCM(outcomes, treatments, confounders; outcome_extra_covariates=outcome_extra_covariates)
