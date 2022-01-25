using CategoricalArrays

struct Query{T <: NamedTuple} 
    name::Union{String, Nothing}
    case::T
    control::T
end

Query(case::NamedTuple, control::NamedTuple; name=nothing) = 
    Query(name, case, control)

Query(;case=NamedTuple{}(), control=NamedTuple{}(), name=nothing) = 
    Query(name, case, control)

variables(query::Query{<:NamedTuple{names}}) where names = names

"""
The order of the variables in the queries and the Treatment Table 
should be the same.
"""
function check_ordering(queries, T)
    Tnames = Tables.columnnames(T)
    for query in queries
        Tnames == variables(query) || 
            throw(ArgumentError, "The variables in T and one of the queries seem to "*
                                "differ, please use the same names. \n Hint: The ordering should match.")
    end
end

"""
    interaction_combinations(query::NamedTuple{names,})
Returns a generator over the different combinations of interactions that
can be built from a query.
"""
interaction_combinations(query::Query) =
    Iterators.product(zip(query.case, query.control)...)


"""
    indicator_fns(query::NamedTuple{names,})

Implements the (-1)^{n-j} formula representing the cross-value of
indicator functions,  where:
    - n is the order of interaction considered
    - j is the number of treatment variables equal to the "case" value
"""
function indicator_fns(query::Query{<:NamedTuple{names, T}}) where {names, T}
    N = length(names)
    indics = ImmutableDict{T, Int}()
    for comb in interaction_combinations(query)
        indics = ImmutableDict(
            indics, 
            comb => (-1)^(N - sum(comb[i] == query.case[i] for i in 1:N))
        )
    end
    return indics
end