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
using LogExpFunctions
using Random
using DifferentiationInterface
using Graphs
using MetaGraphsNext
using Combinatorics
using SplitApplyCombine
using OrderedCollections
using AutoHashEquals
using StatisticalMeasures
using DataFrames
using GLMNet
using ToeplitzMatrices
using ComputationalResources
using Base.Threads
using Printf

# #############################################################################
# EXPORTS
# #############################################################################

export SCM, StaticSCM, add_equations!, add_equation!, parents, vertices
export CM, ATE, AIE
export AVAILABLE_ESTIMANDS
export factorialEstimand, factorialEstimands
export Tmle, Ose, Plugin
export TMLEE, OSE, NAIVE
export JointEstimand, ComposedEstimand
export var, estimate, pvalue, confint, emptyIC
export significance_test, OneSampleTTest, OneSampleZTest, OneSampleHotellingT2Test
export compose
export default_models, with_encoder
export BackdoorAdjustment, identify
export Configuration
export brute_force_ordering, groups_ordering
export gradients, epsilons, estimates
export AdaptiveCorrelationStrategy, GreedyStrategy
export LassoCTMLE
export CausalStratifiedCV, CV, StratifiedCV, Holdout
export CPUThreads, CPU1

# #############################################################################
# INCLUDES
# #############################################################################

include("scm.jl")
include("adjustment.jl")
include("estimands.jl")
include("estimates.jl")
include("utils.jl")
include("estimators.jl")
include("estimand_ordering.jl")
include("resampling.jl")

include("counterfactual_mean_based/estimands.jl")
include("counterfactual_mean_based/estimates.jl")
include("counterfactual_mean_based/fluctuation.jl")
include("counterfactual_mean_based/collaborative_template.jl")
include("counterfactual_mean_based/nuisance_estimators.jl")
include("counterfactual_mean_based/covariate_based_strategies.jl")
include("counterfactual_mean_based/lasso_strategy.jl")
include("counterfactual_mean_based/estimators.jl")
include("counterfactual_mean_based/clever_covariate.jl")
include("counterfactual_mean_based/gradient.jl")

include("configuration.jl")
include("testing.jl")
include("cache.jl")

end
