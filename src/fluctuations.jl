"""
This model is just a thin wrapper around a GLM to be used in A TMLEstimator. 
The query hyperparameter enables that the fit procedure will only re-fit 
this model when the query is changed. Indeed a top level hyper parameter would lead to 
a re-fit of the whole TMLE procedure.

# Arguments
- glm: Union{LinearRegressor, LinearBinaryClassifier},
- query: A NamedTuple defining the reference categories for the targeted step. For isntance, 
query = (col₁=[true, false], col₂=["a", "b"]) defines the interaction 
between col₁ and col₂ where (true, "a") are the `case` categories and (false, "b") are the control categories.
"""
mutable struct Fluctuation <: MLJ.Model
    glm::Union{LinearRegressor, LinearBinaryClassifier}
    query::Union{NamedTuple, Nothing}
    indicators::Union{Dict, Nothing}

    function Fluctuation(glm, query)
        glm.offsetcol = :offset
        glm.fit_intercept = false
        indicators = query isa Nothing ? nothing : indicator_fns(query)
        new(glm, query, indicators)
    end
end

continuousfluctuation(;query=nothing) = Fluctuation(LinearRegressor(), query)
binaryfluctuation(;query=nothing) = Fluctuation(LinearBinaryClassifier(), query)

function Base.setproperty!(model::Fluctuation, name::Symbol, x)
    name == :indicators && throw(ArgumentError("This field must not be changed manually."))
    name != :query && setfield!(model, name, x)

    indicators = indicator_fns(x)
    setfield!(model, :query, x)
    setfield!(model, :indicators, indicators)
end


MLJ.fit(model::Fluctuation, verbosity::Int, X, y) =
    MLJ.fit(model.glm, verbosity, X, y)

MLJ.fitted_params(model::Fluctuation, fitresult) =
    MLJ.fitted_params(model.glm, fitresult)

MLJ.predict_mean(model::Fluctuation, fitresult, Xnew) =
    MLJ.predict_mean(model.glm, fitresult, Xnew)

MLJ.predict(model::Fluctuation, fitresult, Xnew) =
    MLJ.predict(model.glm, fitresult, Xnew)