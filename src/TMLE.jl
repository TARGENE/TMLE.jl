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

# #############################################################################
# OVERLOADED METHODS
# #############################################################################

import MLJBase.fit
import MLJBase.check

# #############################################################################
# EXPORTS
# #############################################################################

export TMLEstimator, Query, Report
export FullCategoricalJoint
export fit
export ztest, pvalue, confint, queryreport, queryreports, briefreport

# #############################################################################
# INCLUDES
# #############################################################################

include("query.jl")
include("model.jl")
include("report.jl")
include("jointmodels.jl")
include("utils.jl")


end
