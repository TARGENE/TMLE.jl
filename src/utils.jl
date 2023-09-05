###############################################################################
## General Utilities
###############################################################################

selectcols(data, cols) = data |> TableOperations.select(cols...) |> Tables.columntable

function logit!(v)
    for i in eachindex(v)
        v[i] = logit(v[i])
    end
end

function nomissing(table)
    sch = Tables.schema(table)
    for type in sch.types
        if nonmissingtype(type) != type
            coltable = table |> 
                       TableOperations.dropmissing |> 
                       Tables.columntable
            return NamedTuple{keys(coltable)}([disallowmissing(col) for col in coltable])
        end
    end
    return table
end

function nomissing(table, columns)
    columns = selectcols(table, columns)
    return nomissing(columns)
end

function indicator_values(indicators, T)
    indic = zeros(Float64, nrows(T))
    for (index, row) in enumerate(Tables.namedtupleiterator(T))
        indic[index] = get(indicators, values(row), 0.)
    end
    return indic
end


expected_value(ŷ::UnivariateFiniteVector{Multiclass{2}}) = pdf.(ŷ, levels(first(ŷ))[2])
expected_value(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = mean.(ŷ)
expected_value(ŷ::AbstractVector{<:Real}) = ŷ

training_expected_value(Q::Machine, dataset) = expected_value(predict(Q, dataset))


function counterfactualTreatment(vals, T)
    Tnames = Tables.columnnames(T)
    n = nrows(T)
    NamedTuple{Tnames}(
            [categorical(repeat([vals[i]], n), levels=levels(Tables.getcolumn(T, name)), ordered=isordered(Tables.getcolumn(T, name)))
                            for (i, name) in enumerate(Tnames)])
end