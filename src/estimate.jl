
abstract type AsymptoticallyLinearEstimate end

struct TMLEstimate{T<:AbstractFloat} <: AsymptoticallyLinearEstimate
    Ψ̂::T
    IC::Vector{T}
end

struct OSEstimate{T<:AbstractFloat} <: AsymptoticallyLinearEstimate
    Ψ̂::T
    IC::Vector{T}
end

struct ComposedEstimate{T<:AbstractFloat} <: AsymptoticallyLinearEstimate
    Ψ̂::Array{T}
    σ̂::Matrix{T}
    n::Int
end

struct TMLEResult{P <: Estimand, T<: AbstractFloat}
    estimand::P
    tmle::TMLEstimate{T}
    onestep::OSEstimate{T}
    initial::T
end

function Base.show(io::IO, ::MIME"text/plain", est::ComposedEstimate)
    if length(est.σ̂) !== 1
        println(io, string("Estimate: ", estimate(est), "\nVariance: \n", var(est)))
    else
        testresult = OneSampleTTest(est)
        data = [estimate(est) confint(testresult) pvalue(testresult);]
        headers = ["Estimate", "95% Confidence Interval", "P-value"]
        pretty_table(io, data;header=headers)
    end
end

    

function Base.show(io::IO, ::MIME"text/plain", est::AsymptoticallyLinearEstimate)
    testresult = OneSampleTTest(est)
    data = [estimate(est) confint(testresult) pvalue(testresult);]
    pretty_table(io, data;header=["Estimate", "95% Confidence Interval", "P-value"])
end

function Base.show(io::IO, ::MIME"text/plain", r::TMLEResult)
    tmletest = OneSampleTTest(r.tmle)
    onesteptest = OneSampleTTest(r.onestep)
    data = [
        :TMLE estimate(tmle(r)) confint(tmletest) pvalue(tmletest);
        :OSE estimate(ose(r)) confint(onesteptest) pvalue(onesteptest);
        :Naive r.initial nothing nothing
        ]
    pretty_table(io, data;header=["Estimator", "Estimate", "95% Confidence Interval", "P-value"])
end

tmle(result::TMLEResult) = result.tmle
ose(result::TMLEResult) = result.onestep
initial(result::TMLEResult) = result.initial

"""
    Distributions.estimate(r::AsymptoticallyLinearEstimate)

Retrieves the final estimate: after the TMLE step.
"""
Distributions.estimate(r::AsymptoticallyLinearEstimate) = r.Ψ̂

"""
    Distributions.estimate(r::ComposedEstimate)

Retrieves the final estimate: after the TMLE step.
"""
Distributions.estimate(r::ComposedEstimate) = length(r.Ψ̂) == 1 ? r.Ψ̂[1] : r.Ψ̂

"""
    var(r::AsymptoticallyLinearEstimate)

Computes the estimated variance associated with the estimate.
"""
Statistics.var(r::AsymptoticallyLinearEstimate) = var(r.IC)/size(r.IC, 1)

"""
    var(r::ComposedEstimate)

Computes the estimated variance associated with the estimate.
"""
Statistics.var(r::ComposedEstimate) = length(r.σ̂) == 1 ? r.σ̂[1] / r.n : r.σ̂ ./ r.n


"""
    OneSampleZTest(r::AsymptoticallyLinearEstimate, Ψ₀=0)

Performs a Z test on the AsymptoticallyLinearEstimate.
"""
HypothesisTests.OneSampleZTest(r::AsymptoticallyLinearEstimate, Ψ₀=0) = OneSampleZTest(estimate(r), std(r.IC), size(r.IC, 1), Ψ₀)

"""
    OneSampleTTest(r::AsymptoticallyLinearEstimate, Ψ₀=0)

Performs a T test on the AsymptoticallyLinearEstimate.
"""
HypothesisTests.OneSampleTTest(r::AsymptoticallyLinearEstimate, Ψ₀=0) = OneSampleTTest(estimate(r), std(r.IC), size(r.IC, 1), Ψ₀)

"""
    OneSampleTTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleTTest(r::ComposedEstimate, Ψ₀=0) 
    @assert length(r.Ψ̂) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleTTest(estimate(r), sqrt(r.σ̂[1]), r.n, Ψ₀)
end

"""
    OneSampleZTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleZTest(r::ComposedEstimate, Ψ₀=0) 
    @assert length(r.Ψ̂) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleZTest(estimate(r), sqrt(r.σ̂[1]), r.n, Ψ₀)
end

"""
    compose(f, estimation_results::Vararg{AsymptoticallyLinearEstimate, N}) where N

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
- estimation_results: 1 or more `AsymptoticallyLinearEstimate` structs.

# Examples

Assuming `res₁` and `res₂` are TMLEs:

```julia
f(x, y) = [x^2 - y, y - 3x]
compose(f, res₁, res₂)
```
"""
function compose(f, estimators::Vararg{AsymptoticallyLinearEstimate, N}; backend=AD.ZygoteBackend()) where N
    Σ = cov(estimators...)
    estimates = [estimate(r) for r in estimators]
    f₀, Js = AD.value_and_jacobian(backend, f, estimates...)
    J = hcat(Js...)
    n = size(first(estimators).IC, 1)
    σ₀ = J*Σ*J'
    return ComposedEstimate(collect(f₀), σ₀, n)
end

function Statistics.cov(estimators::Vararg{AsymptoticallyLinearEstimate, N}) where N
    X = hcat([r.IC for r in estimators]...)
    return Statistics.cov(X, dims=1, corrected=true)
end
