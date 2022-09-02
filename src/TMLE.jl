module TMLE

using Tables
using TableOperations
using CategoricalArrays
using MLJBase
using HypothesisTests
using Base: Iterators, ImmutableDict
using MLJGLMInterface
using MLJModels
using Missings
using Statistics

# #############################################################################
# OVERLOADED METHODS
# #############################################################################

import MLJBase.check

# #############################################################################
# EXPORTS
# #############################################################################

export TMLEstimator, Query, TMLEReport
export FullCategoricalJoint
export fit
export ztest, pvalue, confint, summarize
export MachineReporter, Reporter, JLD2Saver
export ConditionalMean, ATE
export TMLECache
export tmle, tmle!
export var, estimate, OneSampleTTest

# #############################################################################
# INCLUDES
# #############################################################################

include("query.jl")
include("model.jl")
include("report.jl")
include("callbacks.jl")
include("jointmodels.jl")
include("utils.jl")
include("parameters.jl")
include("cache.jl")
include("estimate.jl")

end
