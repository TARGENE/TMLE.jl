module TMLE

using Tables: columnnames
using Distributions: expectation
using Tables
using Distributions
using CategoricalArrays
using GLM
using MLJBase
using MLJ

LinearRegressor = @load LinearRegressor pkg=GLM verbosity=0
LinearBinaryClassifier = @load LinearBinaryClassifier pkg=GLM verbosity=0

# #############################################################################
# OVERLOADED METHODS
# #############################################################################

import MLJ.fit
import MLJBase.check

# #############################################################################
# EXPORTS
# #############################################################################

export ATEEstimator, InteractionATEEstimator
export ContinuousFluctuation, BinaryFluctuation
export fit
export confint, pvalue

# #############################################################################
# INCLUDES
# #############################################################################

include("utils.jl")
include("fluctuations.jl")
include("abstract.jl")
include("ate.jl")
include("interaction_ate.jl")


### Test

mutable struct WrappedRegressor2 <: DeterministicComposite
	regressor
end

# keyword constructor
WrappedRegressor2(; regressor=LinearRegressor()) = WrappedRegressor2(regressor)

function MLJ.fit(model::WrappedRegressor2, verbosity::Integer, X, y)
	Xs = source(X)
	ys = source(y)

	ridge = machine(model.regressor, Xs, ys)
	yhat = MLJ.predict(ridge, Xs)
    target = @node mean(yhat)

	mach = machine(Deterministic(), Xs, ys; predict=target)

	return!(mach, model, verbosity)
end

end
