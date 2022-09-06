module TMLE

using Tables
using TableOperations
using CategoricalArrays
using MLJBase
using HypothesisTests
using Base: Iterators
using MLJGLMInterface
using MLJModels
using Missings
using Statistics
using Zygote

# #############################################################################
# EXPORTS
# #############################################################################

export CM, ATE, IATE
export tmle, tmle!
export var, estimate, OneSampleTTest, pvalue, confint

# #############################################################################
# INCLUDES
# #############################################################################

include("jointmodels.jl")
include("parameters.jl")
include("utils.jl")
include("cache.jl")
include("estimate.jl")

end
