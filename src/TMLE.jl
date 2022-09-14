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

# #############################################################################
# EXPORTS
# #############################################################################

export NuisanceSpec
export CM, ATE, IATE
export tmle, tmle!
export var, cov, estimate, OneSampleTTest, OneSampleZTest, pvalue, confint
export compose

# #############################################################################
# INCLUDES
# #############################################################################

include("treatment_transformer.jl")
include("jointmodels.jl")
include("parameters.jl")
include("utils.jl")
include("cache.jl")
include("estimate.jl")

end
