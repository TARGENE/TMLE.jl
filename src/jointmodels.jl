"""
    FullCategoricalJoint(model)

A thin wrapper around a classifier.
"""
mutable struct FullCategoricalJoint <: Supervised
    model
end


function encode(Y, encoding)
    Yraw = unwrap.(Y)
    categorical([encoding[Tuple(row)] for row in eachrow(Yraw)])
end


function MLJ.fit(model::FullCategoricalJoint, verbosity::Int, X, Y::CategoricalArray)
    # Define the Encoding
    ncols = size(Y)[2]
    joint_levels_it = Base.Iterators.product((levels(Y[:, i]) for i in 1:ncols)...)
    encoding = Dict(Tuple(jl) => i for (i, jl) in enumerate(joint_levels_it))

    # Fit the underlying model
    y_multi = encode(Y, encoding)
    fitresult, cache, report = MLJ.fit(model.model, verbosity, X, y_multi)

    return (encoding=encoding, model_fitresult=fitresult), cache, report
end


MLJ.predict(model::FullCategoricalJoint, fitresult, Xnew) =
    MLJ.predict(model.model, fitresult.model_fitresult, Xnew)


function density(m::Machine{FullCategoricalJoint,}, X, Y)
    ypred = MLJ.predict(m, X)
    y_multi = encode(Y, m.fitresult.encoding)
    pdf.(ypred, y_multi)
end
