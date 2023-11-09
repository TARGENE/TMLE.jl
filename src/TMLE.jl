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
using PrecompileTools
using PrettyTables
using Random
import AbstractDifferentiation as AD
using Graphs
using MetaGraphsNext
using Combinatorics

# #############################################################################
# EXPORTS
# #############################################################################

export SCM, StaticSCM, add_equations!, add_equation!, parents, vertices
export CounterfactualMean, CM
export AverageTreatmentEffect, ATE
export InteractionAverageTreatmentEffect, IATE
export AVAILABLE_ESTIMANDS
export optimize_ordering, optimize_ordering!
export TMLEE, OSE, NAIVE
export var, estimate, OneSampleTTest, OneSampleZTest, pvalue, confint
export compose
export TreatmentTransformer, with_encoder, encoder
export BackdoorAdjustment, identify
export last_fluctuation_epsilon
export estimands_from_yaml, estimands_to_yaml
export to_dict, from_dict!, Configuration
export brute_force_ordering, groups_ordering

# #############################################################################
# INCLUDES
# #############################################################################

include("utils.jl")
include("scm.jl")
include("estimands.jl")
include("estimators.jl")
include("estimates.jl")
include("treatment_transformer.jl")
include("estimand_ordering.jl")

include("counterfactual_mean_based/estimands.jl")
include("counterfactual_mean_based/estimates.jl")
include("counterfactual_mean_based/fluctuation.jl")
include("counterfactual_mean_based/estimators.jl")
include("counterfactual_mean_based/clever_covariate.jl")
include("counterfactual_mean_based/gradient.jl")
include("counterfactual_mean_based/adjustment.jl")

include("configuration.jl")

# #############################################################################
# PRECOMPILATION WORKLOAD
# #############################################################################

function run_precompile_workload()
    @setup_workload begin
        # Putting some things in `@setup_workload` instead of `@compile_workload` can reduce the size of the
        # precompile file and potentially make loading faster.
        n = 1000
        C₁ = rand(n)
        W₁ = rand(n)
        W₂ = rand(n)
        μT₁ = logistic.(1 .+ W₁ .- W₂)
        T₁  = categorical(rand(n) .< μT₁)
        μT₂ = logistic.(1 .+ W₁ .+ 2W₂)
        T₂  = categorical(rand(n) .< μT₂)
        μY  = 1 .+ float(T₁) .+ 2W₂ .- C₁
        Y₁  = μY .+ rand(n)
        Y₂  = categorical(rand(n) .< logistic.(μY))

        dataset = (C₁=C₁, W₁=W₁, W₂=W₂, T₁=T₁, T₂=T₂, Y₁=Y₁, Y₂=Y₂)

        @compile_workload begin
            # SCM constructors
            ## Incremental
            scm = SCM()
            push!(scm, SE(:Y₁, [:T₁, :W₁, :W₂], model=LinearRegressor()))
            push!(scm, SE(:T₁, [:W₁, :W₂]))
            setmodel!(scm.T₁, LinearBinaryClassifier())
            ## Implicit through estimand
            for estimand_type in [CM, ATE, IATE]
                estimand_type(
                    outcome=:Y₁, 
                    treatment=(T₁=true,), 
                    confounders=[:W₁, :W₂],
                    outcome_model=LinearRegressor()
                )
            end
            ## Complete
            # Not using the `with_encoder` for now because it crashes precompilation
            # and fit can still happen somehow.
            scm = SCM(
                SE(:Y₁, [:T₁, :W₁, :W₂], model=LinearRegressor()),
                SE(:T₁, [:W₁, :W₂],model=LinearBinaryClassifier()),
                SE(:Y₂, [:T₁, :T₂, :W₁, :W₂, :C₁], model=LinearBinaryClassifier()),
                SE(:T₂, [:W₁, :W₂],model=LinearBinaryClassifier()),
            )

            # Estimate some parameters
            Ψ₁ = CM(
                scm,
                outcome =:Y₁,
                treatment=(T₁=true,),
            )
            tmle_result₁, fluctuation = tmle!(Ψ₁, dataset, verbosity=0)
            OneSampleTTest(tmle_result₁)
            OneSampleZTest(tmle_result₁)

            ose_result₁, fluctuation = ose!(Ψ₁, dataset, verbosity=0)
            OneSampleTTest(ose_result₁)
            OneSampleZTest(ose_result₁)

            naive_plugin_estimate(Ψ₁)
            # ATE
            Ψ₂ = ATE(
                scm,
                outcome=:Y₂,
                treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false))
            )
            tmle_result₂, fluctuation = tmle!(Ψ₂, dataset, verbosity=0)
            OneSampleTTest(tmle_result₂)
            OneSampleZTest(tmle_result₂)

            ose_result₂, fluctuation = ose!(Ψ₂, dataset, verbosity=0)
            OneSampleTTest(ose_result₂)
            OneSampleZTest(ose_result₂)

            naive_plugin_estimate(Ψ₂)
            # IATE
            Ψ₃ = IATE(
                scm,
                outcome=:Y₂,
                treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false))
            )
            tmle_result₃, fluctuation = tmle!(Ψ₃, dataset, verbosity=0)
            OneSampleTTest(tmle_result₃)
            OneSampleZTest(tmle_result₃)

            ose_result₃, fluctuation = ose!(Ψ₃, dataset, verbosity=0)
            OneSampleTTest(ose_result₃)
            OneSampleZTest(ose_result₃)

            naive_plugin_estimate(Ψ₃)
            # Composition
            composed_result = compose((x,y) -> x - y, tmle_result₂, tmle_result₁)
            OneSampleTTest(composed_result)
            OneSampleZTest(composed_result)
            
        end
    end

end

# run_precompile_workload()

end
