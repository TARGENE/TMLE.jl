"""
    FullCategoricalJoint(model)

A thin wrapper around a classifier to fit a full categorical joint distribution.
"""
mutable struct FullCategoricalJoint <: Supervised
    model
end

"""
    MLJBase.fit(model::FullCategoricalJoint, verbosity::Int, X, Y)

X and Y should respect the Tables.jl interface.
"""
function MLJBase.fit(model::FullCategoricalJoint, verbosity::Int, X, Y)
    # Define the Encoding
    joint_levels_it = Iterators.product((levels(Tables.getcolumn(Y, n)) 
                            for n in Tables.columnnames(Y))...)
    encoding = Dict(Tuple(jl) => i for (i, jl) in enumerate(joint_levels_it))

    # Fit the underlying model
    y_multi = encode(Y, encoding, collect(values(encoding)))
    fitresult, cache, report = MLJBase.fit(model.model, verbosity, X, y_multi)

    return (encoding=encoding, levels=levels(y_multi), model_fitresult=fitresult), cache, report
end


MLJBase.predict(model::FullCategoricalJoint, fitresult, Xnew) =
    MLJBase.predict(model.model, fitresult.model_fitresult, Xnew)


function encode(Y, encoding, levels)
    y_multi = Vector{Int}(undef, nrows(Y))
    for (i, row) in enumerate(Tables.namedtupleiterator(Y))
        y_multi[i] = encoding[values(row)]
    end
    categorical(y_multi; levels=levels)
end

encode(Y, m::Machine{FullCategoricalJoint,}) = 
    encode(Y, m.fitresult.encoding, m.fitresult.levels)
encode(Y::AbstractNode, m::Machine{FullCategoricalJoint,}) =
    node(y -> encode(y, m), Y)


density(ŷ, y) = pdf.(ŷ, y)


function density(m::Machine{FullCategoricalJoint,}, X, Y)
    ŷ = MLJBase.predict(m, X)
    y_multi = encode(Y, m)
    density(ŷ, y_multi)
end

"""
Fallback for classic probablistic models used when 
the treatment is a single variable
"""
function density(m::Machine, X, y)
    ŷ = MLJBase.predict(m, X)
    density(ŷ, adapt(y))
end

