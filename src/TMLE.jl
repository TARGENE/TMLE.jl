module TMLE

using Tables
using CategoricalArrays
using MLJBase
using HypothesisTests
using Base: Iterators, ImmutableDict
using MLJGLMInterface
using MLJModels


# #############################################################################
# OVERLOADED METHODS
# #############################################################################

import MLJBase.fit
import MLJBase.check

# #############################################################################
# EXPORTS
# #############################################################################

export TMLEstimator, Query, QueryReport
export FullCategoricalJoint
export fit
export ztest, pvalue, confint, getqueryreport, briefreport

# #############################################################################
# INCLUDES
# #############################################################################

include("query.jl")
include("model.jl")
include("report.jl")
include("jointmodels.jl")
include("utils.jl")


end
