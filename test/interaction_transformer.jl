using MLJBase
using Combinatorics
using Tables

mutable struct InteractionTransformer <: Static 
    order::Int
    colnames::Union{Nothing, Vector{Symbol}}
end
InteractionTransformer(;order=2, colnames=nothing) = InteractionTransformer(order, colnames)

interactions(columns, order) = 
    collect(Iterators.flatten(combinations(columns, i) for i in 2:order))

actualcolumns(colnames::Nothing, table) = Tables.columnnames(table)

function actualcolumns(colnames::Vector{Symbol}, table)
    diff = setdiff(model.colnames, Tables.columnnames(table))
    diff != [] && throw(ArgumentError(string("Columns ", join([x for x in diff], ", "), " are not in the dataset")))
    return colnames
end

function interaction(columns, variables...)
    .*((Tables.getcolumn(columns, var) for var in variables)...)
end

function MLJBase.transform(model::InteractionTransformer, _, X)
    colnames = actualcolumns(model.colnames, X)
    interactions_ = interactions(colnames, model.order)
    interaction_colnames = Tuple(Symbol(join(inter, "_")) for inter in interactions_)
    columns = Tables.Columns(X)
    interaction_table = NamedTuple{interaction_colnames}([interaction(columns, inter...) for inter in interactions_])
    return merge(Tables.columntable(X), interaction_table)
end