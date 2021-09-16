"""
    FullCategoricalJoint(model)

A thin wrapper around a classifier.
"""
mutable struct FullCategoricalJoint <: Supervised
    model
end


function encode(Y, encoding; levels=nothing)
    y_multi = Vector{Int}(undef, nrows(Y))
    for (i, row) in enumerate(Tables.namedtupleiterator(Y))
        y_multi[i] = encoding[values(row)]
    end
    categorical(y_multi;levels=levels)
end


function MLJ.fit(model::FullCategoricalJoint, verbosity::Int, X, Y)
    # Define the Encoding
    joint_levels_it = Iterators.product((levels(Tables.getcolumn(Y, n)) 
                            for n in Tables.columnnames(Y))...)
    encoding = Dict(Tuple(jl) => i for (i, jl) in enumerate(joint_levels_it))

    # Fit the underlying model
    y_multi = encode(Y, encoding)
    fitresult, cache, report = MLJ.fit(model.model, verbosity, X, y_multi)

    return (encoding=encoding, levels=levels(y_multi), model_fitresult=fitresult), cache, report
end


MLJ.predict(model::FullCategoricalJoint, fitresult, Xnew) =
    MLJ.predict(model.model, fitresult.model_fitresult, Xnew)


function density(m::Machine{FullCategoricalJoint,}, X, Y)
    ypred = MLJ.predict(m, X)
    y_multi = encode(Y, m.fitresult.encoding;levels=m.fitresult.levels)
    pdf.(ypred, y_multi)
end
