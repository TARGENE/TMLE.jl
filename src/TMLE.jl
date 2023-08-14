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
using YAML
using PrecompileTools
using PrettyTables
using Random
import AbstractDifferentiation as AD

# #############################################################################
# EXPORTS
# #############################################################################

export SE, StructuralEquation
export StructuralCausalModel, SCM, StaticConfoundedModel
export setmodel!, equations, reset!, parents
export ConditionalMean, CM
export AverageTreatmentEffect, ATE
export InteractionAverageTreatmentEffect, IATE
export AVAILABLE_ESTIMANDS
export fit!, optimize_ordering, optimize_ordering!
export tmle!, tmle, ose, initial
export var, estimate, OneSampleTTest, OneSampleZTest, pvalue, confint
export compose
export TreatmentTransformer, with_encoder
export BackdoorAdjustment

# #############################################################################
# INCLUDES
# #############################################################################

include("scm.jl")
include("estimands.jl")
include("adjustment.jl")
include("utils.jl")
include("estimation.jl")
include("estimate.jl")
include("treatment_transformer.jl")


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
                estimand_type(outcome=:Y₁, treatment=(T₁=true,), confounders=[:W₁, :W₂])
            end
            ## Complete
            scm = SCM(
                SE(:Y₁, [:T₁, :W₁, :W₂], model=with_encoder(LinearRegressor())),
                SE(:T₁, [:W₁, :W₂],model=LinearBinaryClassifier()),
                SE(:Y₂, [:T₁, :T₂, :W₁, :W₂, :C₁], model=with_encoder(LinearBinaryClassifier())),
                SE(:T₂, [:W₁, :W₂],model=LinearBinaryClassifier()),
            )

            # Estimate some parameters
            Ψ₁ = CM(
                scm,
                outcome =:Y₁,
                treatment=(T₁=true,),
            )
            result₁, fluctuation = tmle!(Ψ₁, dataset)

            Ψ₂ = ATE(
                scm,
                outcome=:Y₂,
                treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false))
            )
            result₂, fluctuation = tmle!(Ψ₂, dataset)

            Ψ₃ = IATE(
                scm,
                outcome=:Y₂,
                treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false))
            )
            result₃, fluctuation = tmle!(Ψ₃, dataset)

            # Composition
            composed_result = compose((x,y) -> x - y, tmle(result₂), tmle(result₁))

            # Results manipulation
            initial(result)
            OneSampleTTest(tmle(result₃))
            OneSampleZTest(tmle(result₃))
            OneSampleTTest(ose(result₃))
            OneSampleZTest(ose(result₃))
            OneSampleZTest(composed_result)
            
        end
    end

end

# run_precompile_workload()

end
