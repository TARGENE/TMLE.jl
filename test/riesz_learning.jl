module TestRieszNet

using Test
using TMLE
using RieszLearning
using Random
using StableRNGs
using Distributions
using LogExpFunctions
using CategoricalArrays
using DataFrames

TEST_DIR = joinpath(pkgdir(TMLE), "test")

include(joinpath(TEST_DIR, "counterfactual_mean_based", "ate_simulations.jl"))


riesznet = RieszNetModel()

Ψ = ATE(
    outcome = :Y,
    treatment_values = (T=(case=1, control=0),),
    treatment_confounders = (T = [:W₁, :W₂, :W₃],)
)
dataset, Ψ₀ = continuous_outcome_binary_treatment_pb()

@testset "Test RieszRepresenterEstimator" begin
    
end

end

true