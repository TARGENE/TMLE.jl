module GraphsTMLEExt

using TMLE
using Graphs
using CairoMakie
using GraphMakie

function edges_list_and_nodes_mapping(scm::SCM)
    edge_list = []
    node_ids = Dict()
    node_id = 1
    for (outcome, eq) in equations(scm)
        if outcome ∉ keys(node_ids)
            push!(node_ids, outcome => node_id)
            node_id += 1
        end
        outcome_id = node_ids[outcome]
        for parent in TMLE.parents(eq)
            if parent ∉ keys(node_ids)
                push!(node_ids, parent => node_id)
                node_id += 1
            end
            parent_id = node_ids[parent]
            push!(edge_list, Edge(parent_id, outcome_id))
        end
    end
    return edge_list, node_ids 
end

function DAG(scm::SCM)
    edges_list, nodes_mapping = edges_list_and_nodes_mapping(scm)
    return SimpleDiGraphFromIterator(edges_list), nodes_mapping
end

function graphplot(scm::SCM; kwargs...)
    graph, nodes_mapping = DAG(scm)
    f, ax, p = graphplot(graph;
        ilabels = [x[1] for x in sort(collect(nodes_mapping), by=x -> x[2])],
        kwargs...)
    Legend(
        f[1, 2], 
        [MarkerElement(color=:black, marker='⋆', markersize = 10) for eq in equations(scm)], 
        [TMLE.string_repr(eq) for (_, eq) in equations(scm)])
    hidedecorations!(ax);
    hidespines!(ax); 
    ax.aspect = DataAspect()
    return fig, ax, p
end

end