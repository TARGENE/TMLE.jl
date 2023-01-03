
abstract type AbstractTMLE end

struct PointTMLE{T<:AbstractFloat} <: AbstractTMLE
    Ψ̂::T
    IC::Vector{T}
    Ψ̂ᵢ::T
end

struct ComposedTMLE{T<:AbstractFloat} <: AbstractTMLE
    Ψ̂::Array{T}
    σ̂::Matrix{T}
end

initial_estimate(r::PointTMLE) = r.Ψ̂ᵢ
estimate(r::PointTMLE) = r.Ψ̂
estimate(r::ComposedTMLE) = length(r.Ψ̂) == 1 ? r.Ψ̂[1] : r.Ψ̂

Statistics.var(r::PointTMLE) = var(r.IC)/size(r.IC, 1)
Statistics.var(r::ComposedTMLE) = length(r.σ̂) == 1 ? r.σ̂[1] : r.σ̂

HypothesisTests.OneSampleZTest(r::PointTMLE, Ψ₀=0) = OneSampleZTest(estimate(r), std(r.IC), size(r.IC, 1), Ψ₀)


HypothesisTests.OneSampleTTest(r::PointTMLE, Ψ₀=0) = OneSampleTTest(estimate(r), std(r.IC), size(r.IC, 1), Ψ₀)
function HypothesisTests.OneSampleTTest(r::ComposedTMLE, Ψ₀=0) 
    @assert length(r.Ψ̂) > 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleTTest(estimate(r), sqrt(var(r)), 1, Ψ₀)
end


"""
    compose(f, estimation_results::Vararg{PointTMLE, N}) where N

Provides an estimator of f(estimation_results...).

# Mathematical details

The following is a summary from `Asymptotic Statistics`, A. W. van der Vaart.

Consider k TMLEs computed from a dataset of size n and embodied by Tₙ = (T₁,ₙ, ..., Tₖ,ₙ). 
Since each of them is asymptotically normal, the multivariate CLT provides the joint 
distribution:

    √n(Tₙ - Ψ₀) ↝ N(0, Σ), 
    
where Σ is the covariance matrix of the TMLEs influence curves.

Let f:ℜᵏ→ℜᵐ, be a differentiable map at Ψ₀. Then, the delta method provides the
limiting distribution of √n(f(Tₙ) - f(Ψ₀)). Because Tₙ is normal, the result is:

    √n(f(Tₙ) - f(Ψ₀)) ↝ N(0, ∇f(Ψ₀) ̇Σ ̇(∇f(Ψ₀))ᵀ),

where ∇f(Ψ₀):ℜᵏ→ℜᵐ is a linear map such that by abusing notations and identifying the 
function with the multiplication matrix: ∇f(Ψ₀):h ↦ ∇f(Ψ₀) ̇h. And the matrix ∇f(Ψ₀) is 
the jacobian of f at Ψ₀.

Hence, the only thing we need to do is:
- Compute the covariance matrix Σ
- Compute the jacobian ∇f, which can be done using Julia's automatic differentiation facilities.
- The final estimator is normal with mean f₀=f(Ψ₀) and variance σ₀=∇f(Ψ₀) ̇Σ ̇(∇f(Ψ₀))ᵀ

# Arguments

- f: An array-input differentiable map.
- estimation_results: 1 or more `PointTMLE` structs.

# Examples

Assuming `res₁` and `res₂` are TMLEs:

```julia
f(x, y) = [x^2 - y, y - 3x]
compose(f, res₁, res₂)
```
"""
function compose(f, estimators::Vararg{PointTMLE, N}; backend=AD.ZygoteBackend()) where N
    Σ = cov(estimators...)
    estimates = [estimate(r) for r in estimators]
    f₀, Js = AD.value_and_jacobian(backend, f, estimates...)
    J = hcat(Js...)
    n = size(first(estimators).IC, 1)
    σ₀ = (J*Σ*J')/n
    return ComposedTMLE(collect(f₀), σ₀)
end

function Statistics.cov(estimators::Vararg{PointTMLE, N}) where N
    X = hcat([r.IC for r in estimators]...)
    return Statistics.cov(X, dims=1, corrected=true)
end
