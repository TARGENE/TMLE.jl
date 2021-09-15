module TMLE

using Tables: columnnames
using Distributions: expectation
using Tables
using Distributions
using CategoricalArrays
using GLM
using MLJBase
using MLJ
using Base: Iterators

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
export FullCategoricalJoint
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
include("jointmodels.jl")

end
