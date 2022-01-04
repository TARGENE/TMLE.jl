module TMLE

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
import Distributions.estimate
import Distributions.stderror

# #############################################################################
# EXPORTS
# #############################################################################

export TMLEstimator
export FullCategoricalJoint
export fit
export summaries

# #############################################################################
# INCLUDES
# #############################################################################

include("model.jl")
include("report.jl")
include("jointmodels.jl")
include("utils.jl")

end
