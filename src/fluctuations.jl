"""
This model is just a thin wrapper around a GLM to be used in A TMLE. 
The query hyperparameter enables that the fit procedure will only re-fit 
this model when the query is changed. Indeed a top level hyper parameter would lead to 
a re-fit of the whole TMLE procedure.

"""
mutable struct Fluctuation <: MLJ.Model
    glm
    query

    function Fluctuation(glm, query)
        glm.offsetcol = :offset
        new(glm, query)
    end
end


function ContinuousFluctuation(;
    glm=LinearRegressor(;fit_intercept=false),
    query=nothing)
    Fluctuation(glm, query)
end


function BinaryFluctuation(;
    glm=LinearBinaryClassifier(fit_intercept=false),
    query=nothing
    )
    Fluctuation(glm, query)
end


MLJ.fit(model::Fluctuation, verbosity::Int, X, y) =
    MLJ.fit(model.glm, verbosity, X, y)

MLJ.fitted_params(model::Fluctuation, fitresult) =
    MLJ.fitted_params(model.glm, fitresult)

MLJ.predict_mean(model::Fluctuation, fitresult, Xnew) =
    MLJ.predict_mean(model.glm, fitresult, Xnew)

MLJ.predict(model::Fluctuation, fitresult, Xnew) =
    MLJ.predict(model.glm, fitresult, Xnew)
