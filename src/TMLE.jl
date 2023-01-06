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
using Distributions
using Zygote
using AbstractDifferentiation
using LogExpFunctions
using YAML

# #############################################################################
# EXPORTS
# #############################################################################

export NuisanceSpec, TMLECache, CM, ATE, IATE
export tmle, tmle!
export var, estimate, initial_estimate, OneSampleTTest, OneSampleZTest, pvalue, confint
export compose
export parameters_from_yaml

# #############################################################################
# INCLUDES
# #############################################################################

include("treatment_transformer.jl")
include("parameters.jl")
include("utils.jl")
include("cache.jl")
include("estimate.jl")

end
