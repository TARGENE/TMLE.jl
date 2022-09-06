
abstract type AbstractEstimationResult end

struct EstimationResult <: AbstractEstimationResult
    Ψ::Parameter
    Ψ̂::AbstractFloat
    IC::AbstractVector
end

struct ComposedEstimationResult <: AbstractEstimationResult
    f::Function
    estimation_results
    Ψ̂::AbstractFloat
    variance::AbstractFloat
end

estimate(r::AbstractEstimationResult) = r.Ψ̂

Statistics.var(r::EstimationResult) = var(r.IC)/size(r.IC, 1)
Statistics.var(r::ComposedEstimationResult) = r.variance

HypothesisTests.OneSampleTTest(r::EstimationResult, Ψ₀=0) = OneSampleTTest(r.Ψ̂, std(r.IC), size(r.IC, 1), Ψ₀)
HypothesisTests.OneSampleTTest(r::ComposedEstimationResult, Ψ₀=0) = OneSampleTTest(r.Ψ̂, sqrt(var(r)), 1, Ψ₀)


function compose(f, estimation_results...)
    estimates = (estimate(r) for r in estimation_results)
    Ψ̂ = f(estimates...)
    Σ = multivariateCovariance((r.IC for r in estimation_results)...)
    ∇f = collect(Zygote.gradient(f, estimates...))
    return ComposedEstimationResult(f, estimation_results, Ψ̂, ∇f'*Σ*∇f)
end

function multivariateCovariance(args...)
    D = hcat(args...)
    n = size(D, 1)
    return D'D/n
end
