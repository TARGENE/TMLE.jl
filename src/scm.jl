#####################################################################
###                  Structural Equation                          ###
#####################################################################

"""
A SCM is simply a wrapper around a MetaGraph over a Directed Acyclic Graph.
"""
struct SCM
    graph::MetaGraph
    
    function SCM(equations)
        graph =  MetaGraph(
            SimpleDiGraph();
            label_type=Symbol,
            vertex_data_type=Nothing,
            edge_data_type=Nothing
        )
        scm = new(graph)
        add_equations!(scm, equations...)
        return scm
    end
end

SCM(;equations=()) = SCM(equations)

function string_repr(scm::SCM)
    string_rep = "SCM\n---"
    for (vertex_id, vertex_label) in scm.graph.vertex_labels
        vertex_parents = parents(scm, vertex_label)
        if length(vertex_parents) > 0
            digits = split(string(vertex_id), "")
            subscript = join(Meta.parse("'\\U0208$digit'") for digit in digits)
            eq_string = string("\n", vertex_label, " = f", subscript, "(", join(vertex_parents, ", "), ")")
            string_rep = string(string_rep, eq_string)
        end
    end
    return string_rep
end

Base.show(io::IO, ::MIME"text/plain", scm::SCM) =
    println(io, string_repr(scm))

split_outcome_parent_pair(outcome_parents_pair::Pair) = outcome_parents_pair
split_outcome_parent_pair(outcome_parents_pair::AbstractDict{T, Any}) where T = outcome_parents_pair[T(:outcome)], outcome_parents_pair[T(:parents)] 

function add_equations!(scm::SCM, equations...)
    for outcome_parents_pair in equations
        add_equation!(scm, outcome_parents_pair)
    end
end

function add_equation!(scm::SCM, outcome_parents_pair)
    outcome, parents = split_outcome_parent_pair(outcome_parents_pair)
    outcome_symbol = Symbol(outcome)
    add_vertex!(scm.graph, outcome_symbol)
    for parent in parents
        parent_symbol = Symbol(parent)
        add_vertex!(scm.graph, parent_symbol)
        add_edge!(scm.graph, parent_symbol, outcome_symbol)
    end
end

function parents(scm::SCM, label)
    code, _ = scm.graph.vertex_properties[label]
    return Set((scm.graph.vertex_labels[parent_code] for parent_code in scm.graph.graph.badjlist[code]))
end

vertices(scm::SCM) = collect(keys(scm.graph.vertex_properties))

"""
A plate Structural Causal Model where:

- For all outcomes: oᵢ = fᵢ(treatments, confounders, outcome_extra_covariates)
- For all treatments: tⱼ = fⱼ(confounders)

#  Example

StaticSCM([:Y], [:T₁, :T₂], [:W₁, :W₂, :W₃]; outcome_extra_covariates=[:C])
"""
function StaticSCM(outcomes, treatments, confounders)
    outcome_equations = (outcome => unique(vcat(treatments, confounders)) for outcome in outcomes)
    treatment_equations = (treatment => unique(confounders) for treatment in treatments)
    return SCM(equations=(outcome_equations..., treatment_equations...))
end

StaticSCM(;outcomes, treatments, confounders) = 
    StaticSCM(outcomes, treatments, confounders)


to_dict(scm::SCM) = Dict(
    :type => "SCM",
    :equations => [Dict(:outcome => label, :parents => collect(parents(scm, label))) for label ∈ vertices(scm)]
)