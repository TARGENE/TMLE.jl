module TestGradient

using TMLE
using Test
using MLJBase
using StableRNGs
using LogExpFunctions
using Distributions
using MLJLinearModels
using MLJModels
using DataFrames

μY(T, W)  = 1 .+ 2T .- W.*T

function one_treatment_dataset(;n=100)
    rng = StableRNG(123)
    W   = rand(rng, n)
    μT  = logistic.(1 .- 3W)
    T   = rand(rng, n) .< μT
    Y   = μY(T, W) .+ rand(rng, Normal(0, 0.1), n)
    return DataFrame(
        T = categorical(T, ordered=true),
        Y = Y,
        W = W
    )
end

"""
    brute_ccw_cluster_ic(IC_full, y, q0)
    An extension of the CCW cluster IC function that returns information about
    how the clusters/blocks are partitioned.
"""
function expected_cluster_ic_intJ(IC_full, y, q0)
    idx_case = findall(y .== 1)
    idx_ctl  = findall(y .== 0)
    nC, nCo = length(idx_case), length(idx_ctl)
    @assert nC > 0
    @assert nCo > 0
    J = nCo ÷ nC
    @assert J > 0
    used_controls = idx_ctl[1:(J*nC)]
    ic = similar(idx_case, Float64)
    for (i, case_idx) in enumerate(idx_case)
        block = used_controls[(J*(i-1)+1):(J*i)]
        ctl_mean = mean(IC_full[block])
        ic[i] = q0 * IC_full[case_idx] + (1 - q0) * ctl_mean
    end
    return ic, used_controls, J
end

@testset "Test gradient_and_plugin_estimate" begin
    ps_lowerbound = 1e-8
    Ψ = ATE(
        outcome = :Y, 
        treatment_values = (T=(case=1, control=0),), 
        treatment_confounders = (T=[:W],), 
    )
    dataset = one_treatment_dataset(;n=100)
    η = TMLE.CMRelevantFactors(
        TMLE.ConditionalDistribution(:Y, [:T, :W]),
        TMLE.ConditionalDistribution(:T, [:W])
    )
    η̂ = TMLE.CMRelevantFactorsEstimator(
        models=Dict(
            :Y => with_encoder(InteractionTransformer(order=2) |> LinearRegressor()), 
            :T => LogisticClassifier())
    )
    η̂ₙ = η̂(η, dataset, verbosity = 0)
    # Retrieve conditional distributions and fitted_params
    Q = η̂ₙ.outcome_mean
    G = η̂ₙ.propensity_score
    linear_model = fitted_params(Q.machine).deterministic_pipeline.linear_regressor
    intercept = linear_model.intercept
    coefs = Dict(linear_model.coefs)
    # Counterfactual aggregate
    ctf_agg = TMLE.counterfactual_aggregate(Ψ, η̂ₙ.outcome_mean, dataset)
    expected_ctf_agg = (intercept .+ coefs[:T] .+ dataset.W.*coefs[:W] .+ dataset.W.*coefs[:T_W]) .- (intercept .+ dataset.W.*coefs[:W])
    @test ctf_agg ≈ expected_ctf_agg atol=1e-10
    # Gradient Y|X
    ps_machine = only(G.components).machine
    H = 1 ./ pdf.(predict(ps_machine), dataset.T) .* [t == 1 ? 1. : -1. for t in dataset.T]
    expected_∇YX = H .* (dataset.Y .- predict(Q.machine))
    ∇YX = TMLE.∇YX(Ψ, Q, G, dataset; ps_lowerbound=ps_lowerbound)
    @test expected_∇YX == ∇YX
    # Gradient W
    expectedΨ̂ = mean(ctf_agg)
    ∇W = TMLE.∇W(ctf_agg, expectedΨ̂)
    @test ∇W == ctf_agg .- expectedΨ̂
    # gradient_and_plugin_estimate
    IC, Ψ̂ = TMLE.gradient_and_plugin_estimate(Ψ, η̂ₙ, dataset; ps_lowerbound=ps_lowerbound)
    @test expectedΨ̂ == Ψ̂
    @test IC == ∇YX .+ ∇W
end

@testset "ccw_cluster_ic: reduce gradient" begin
    full_IC = [1, 2, 3, 4, 5, 6]
    y = [1, 0, 1, 0, 0, 0]  # 2 cases, 4 controls -> J = 2
    q0 = 0.05
    ic = TMLE.ccw_cluster_ic(full_IC, y, q0)
    # Manual expectation:
    # Case indices: 1, 3; control indices used: first 4 controls = 2,4,5,6; partition -> (2,4) and (5,6)
    exp1 = q0*1 + (1-q0)*mean([2,4])
    exp2 = q0*3 + (1-q0)*mean([5,6])
    @test ic ≈ [exp1, exp2] atol=1e-12
end

@testset "ccw_cluster_ic: exact divisibility" begin
    # 4 cases, 12 controls -> J = 3
    y = vcat(ones(Int,4), zeros(Int,12))
    IC = collect(1.0:length(y))
    q0 = 0.2
    ic = TMLE.ccw_cluster_ic(IC, y, q0)
    expected_ic, used, J = expected_cluster_ic_intJ(IC, y, q0)
    @test J == 3
    @test ic ≈ expected_ic atol=1e-12
    # All controls used
    @test length(used) == 12
end

@testset "ccw_cluster_ic: non-divisible controls (truncation)" begin
    # 5 cases, 16 controls -> J = floor(16/5)=3, leftover=1 control dropped
    y = vcat(ones(Int,5), zeros(Int,16))
    IC = randn(length(y))
    q0 = 0.13
    ic = TMLE.ccw_cluster_ic(IC, y, q0)
    expected_ic, used, J = expected_cluster_ic_intJ(IC, y, q0)
    @test J == 3
    @test length(used) == 15  # 5 * 3
    @test ic ≈ expected_ic atol=1e-12
    # Dropped control index is the last control
    dropped = setdiff(findall(y .== 0), used)
    @test length(dropped) == 1
end

@testset "ccw_cluster_ic: single case uses all controls" begin
    y = vcat(1, zeros(Int,7))  # 1 case, 7 controls -> J=7
    IC = randn(length(y))
    q0 = 0.05
    ic = TMLE.ccw_cluster_ic(IC, y, q0)
    ctl_mean = mean(IC[2:end])
    @test length(ic) == 1
    @test ic[1] ≈ q0*IC[1] + (1-q0)*ctl_mean
end

@testset "ccw_cluster_ic: errors" begin
    IC = randn(5)
    @test_throws DivideError TMLE.ccw_cluster_ic(IC, zeros(Int,5), 0.1)  # no cases
    @test_throws ArgumentError TMLE.ccw_cluster_ic(IC, ones(Int,5), 0.1)   # no controls
end



end

true