
struct EstimationResult
    Ψ::Parameter
    Ψ̂::AbstractFloat
    IC::AbstractVector
end

estimate(r::EstimationResult) = r.Ψ̂

Statistics.var(r::EstimationResult) = var(r.IC)/size(r.IC, 1)
HypothesisTests.OneSampleTTest(r::EstimationResult, Ψ₀=0) = OneSampleTTest(r.Ψ̂, std(r.IC), size(r.IC, 1), Ψ₀)

function compose(f, tmleresports...)
    Σ = multivariateCovariance((tlr.eic for tlr in tmleresports)...)
    ∇f = collect(gradient(f, (tlr.estimate for tlr in tmleresports)...))
    return ∇f'*Σ*∇f
end

function multivariateCovariance(args...)
    D = hcat(args...)
    n = size(D, 1)
    return D'D/n
end
