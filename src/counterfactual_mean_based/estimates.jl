#####################################################################
###                       MLCMRelevantFactors                     ###
#####################################################################

"""
Holds a Sample Split Machine Learning set of estimates (outcome mean, propensity score) 
for counterfactual mean based estimands' relevant factors.
"""
struct MLCMRelevantFactors <: Estimate
    estimand::CMRelevantFactors
    outcome_mean::ConditionalDistributionEstimate
    propensity_score::Tuple{Vararg{ConditionalDistributionEstimate}}
end

string_repr(estimate::MLCMRelevantFactors) = string(
    "Composite Factor Estimate: \n",
    "-------------------------\n- ",
    string_repr(estimate.outcome_mean),"\n- ", 
    join((string_repr(f) for f in estimate.propensity_score), "\n- ")
)

#####################################################################
###                   One Dimensional Estimates                   ###
#####################################################################

struct TMLEstimate{T<:AbstractFloat} <: Estimate
    estimand::StatisticalCMCompositeEstimand
    estimate::T
    std::T
    n::Int
    IC::Vector{T}
end

TMLEstimate(;estimand, estimate::T, std::T, n, IC) where T = TMLEstimate(estimand, estimate, std, n, convert(Vector{T}, IC))

struct OSEstimate{T<:AbstractFloat} <: Estimate
    estimand::StatisticalCMCompositeEstimand
    estimate::T
    std::T
    n::Int
    IC::Vector{T}
end

OSEstimate(;estimand, estimate::T, std::T, n, IC) where T = OSEstimate(estimand, estimate, std, n, convert(Vector{T}, IC))

const EICEstimate = Union{TMLEstimate, OSEstimate}

function to_dict(estimate::T) where T <: EICEstimate
    Dict(
        :type => replace(string(Base.typename(T).wrapper), "TMLE." => ""),
        :estimate => estimate.estimate,
        :estimand => to_dict(estimate.estimand),
        :std => estimate.std,
        :n => estimate.n,
        :IC => estimate.IC
    )
end

emptyIC(estimate::T) where T <: EICEstimate = 
    T(estimate.estimand, estimate.estimate, estimate.std, estimate.n, [])

function Base.show(io::IO, ::MIME"text/plain", est::EICEstimate)
    testresult = OneSampleTTest(est)
    data = [estimate(est) confint(testresult) pvalue(testresult);]
    pretty_table(io, data;header=["Estimate", "95% Confidence Interval", "P-value"])
end

struct ComposedEstimate{T<:AbstractFloat} <: Estimate
    estimate::Array{T}
    σ̂::Matrix{T}
    n::Int
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

"""
    Distributions.estimate(r::EICEstimate)

Retrieves the final estimate: after the TMLE step.
"""
Distributions.estimate(est::EICEstimate) = est.estimate

"""
    Distributions.estimate(r::ComposedEstimate)

Retrieves the final estimate: after the TMLE step.
"""
Distributions.estimate(est::ComposedEstimate) = 
    length(est.estimate) == 1 ? est.estimate[1] : est.estimate

"""
    var(r::EICEstimate)

Computes the estimated variance associated with the estimate.
"""
Statistics.var(est::EICEstimate) = 
    var(est.IC)/size(est.IC, 1)

"""
    var(r::ComposedEstimate)

Computes the estimated variance associated with the estimate.
"""
Statistics.var(est::ComposedEstimate) = 
    length(est.σ̂) == 1 ? est.σ̂[1] / est.n : est.σ̂ ./ est.n


"""
    OneSampleZTest(r::EICEstimate, Ψ₀=0)

Performs a Z test on the EICEstimate.
"""
HypothesisTests.OneSampleZTest(est::EICEstimate, Ψ₀=0) = 
    OneSampleZTest(est.estimate, est.std, est.n, Ψ₀)

"""
    OneSampleTTest(r::EICEstimate, Ψ₀=0)

Performs a T test on the EICEstimate.
"""
HypothesisTests.OneSampleTTest(est::EICEstimate, Ψ₀=0) = 
    OneSampleTTest(est.estimate, est.std, est.n, Ψ₀)

"""
    OneSampleTTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleTTest(est::ComposedEstimate, Ψ₀=0) 
    @assert length(est.estimate) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleTTest(estimate(est), sqrt(est.σ̂[1]), est.n, Ψ₀)
end

"""
    OneSampleZTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleZTest(est::ComposedEstimate, Ψ₀=0) 
    @assert length(est.estimate) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleZTest(estimate(est), sqrt(est.σ̂[1]), est.n, Ψ₀)
end

"""
    compose(f, estimation_results::Vararg{EICEstimate, N}) where N

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
- estimation_results: 1 or more `EICEstimate` structs.

# Examples

Assuming `res₁` and `res₂` are TMLEs:

```julia
f(x, y) = [x^2 - y, y - 3x]
compose(f, res₁, res₂)
```
"""
function compose(f, estimators::Vararg{EICEstimate, N}; backend=AD.ZygoteBackend()) where N
    Σ = cov(estimators...)
    estimates = [estimate(r) for r in estimators]
    f₀, Js = AD.value_and_jacobian(backend, f, estimates...)
    J = hcat(Js...)
    n = size(first(estimators).IC, 1)
    σ₀ = J*Σ*J'
    return ComposedEstimate(collect(f₀), σ₀, n)
end

function Statistics.cov(estimators::Vararg{EICEstimate, N}) where N
    X = hcat([r.IC for r in estimators]...)
    return Statistics.cov(X, dims=1, corrected=true)
end
