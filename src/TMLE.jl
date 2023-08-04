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
using Configurations
using PrecompileTools
using PrettyTables
using Random
import AbstractDifferentiation as AD

# #############################################################################
# EXPORTS
# #############################################################################

export SE, StructuralEquation
export StructuralCausalModel, SCM, StaticConfoundedModel 
export NuisanceSpec, TMLECache, update!, CM, ATE, IATE, isidentified
export treatments, outcome, confounders, parents, fit!, reset!, \
    isidentified, equations
export tmle, tmle!
export var, estimate, initial_estimate, OneSampleTTest, OneSampleZTest, pvalue, confint
export compose
export estimands_from_yaml, estimands_to_yaml, optimize_ordering, optimize_ordering!
export TreatmentTransformer

# #############################################################################
# INCLUDES
# #############################################################################

include("scm.jl")
include("treatment_transformer.jl")
include("estimands.jl")
include("utils.jl")
include("cache.jl")
include("estimate.jl")
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

        dataset = (C₁=C₁, W₁=W₁, W₂=W₂, T₁=T₁, T₂=T₂, Y₁=Y₁, Y₂)

        @compile_workload begin
            # all calls in this block will be precompiled, regardless of whether
            # they belong to your package or not (on Julia 1.8 and higher)
            Ψ = CM(
                outcome =:Y₁,
                treatment=(T₁=true,),
                confounders=[:W₁, :W₂],
                covariates = [:C₁]
            )
            η_spec = NuisanceSpec(
                LinearRegressor(),
                LinearBinaryClassifier()
            )
            r, cache = tmle(Ψ, η_spec, dataset, verbosity=0)
            continuous_estimands = [
                ATE(
                    outcome =:Y₁,
                    treatment=(T₁=(case=true, control=false),),
                    confounders=[:W₁, :W₂],
                    covariates = [:C₁]
                ),
                # IATE(
                #     outcome =:Y₁,
                #     treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
                #     confounders=[:W₁, :W₂],
                #     covariates = [:C₁]
                # )
            ]
            cache = TMLECache(dataset)
            for Ψ in continuous_estimands
                tmle!(cache, Ψ, η_spec, verbosity=0)
            end

            # Precompiling with a binary outcome
            binary_estimands = [
                ATE(
                    outcome =:Y₂,
                    treatment=(T₁=(case=true, control=false),),
                    confounders=[:W₁, :W₂],
                    covariates = [:C₁]
                ),
                # IATE(
                #     outcome =:Y₂,
                #     treatment=(T₁=(case=true, control=false), T₂=(case=true, control=false)),
                #     confounders=[:W₁, :W₂],
                #     covariates = [:C₁]
                # )
            ]
            η_spec = NuisanceSpec(
                LinearBinaryClassifier(),
                LinearBinaryClassifier()
            )

            for Ψ in binary_estimands
                tmle!(cache, Ψ, η_spec, verbosity=0)
            end

        end
    end

end

# run_precompile_workload()

end
