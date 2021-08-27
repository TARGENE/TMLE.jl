module TMLE

using Tables: columnnames
using Distributions: expectation
using Tables
using Distributions
using CategoricalArrays
using GLM
using MLJBase

# #############################################################################
# OVERLOADED METHODS
# #############################################################################

import MLJBase.fit
import MLJBase.check

# #############################################################################
# EXPORTS
# #############################################################################

export ATEEstimator, InteractionATEEstimator
export fit
export confint, pvalue

# #############################################################################
# INCLUDES
# #############################################################################

include("utils.jl")
include("abstract.jl")
include("ate.jl")
include("interaction_ate.jl")

end
