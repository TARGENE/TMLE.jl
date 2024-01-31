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

emptyIC(estimate::T, ::Nothing) where T <: EICEstimate = 
    T(estimate.estimand, estimate.estimate, estimate.std, estimate.n, [])

function emptyIC(estimate::T, pval_threshold::Float64) where T <: EICEstimate
    pval = pvalue(OneSampleZTest(estimate))
    return pval < pval_threshold ? estimate : emptyIC(estimate, nothing)
end

emptyIC(estimate; pval_threshold=nothing) = emptyIC(estimate, pval_threshold)

"""
    Distributions.estimate(r::EICEstimate)

Retrieves the final estimate: after the TMLE step.
"""
Distributions.estimate(est::EICEstimate) = est.estimate

"""
    var(r::EICEstimate)

Computes the estimated variance associated with the estimate.
"""
Statistics.var(est::EICEstimate) = 
    var(est.IC)/size(est.IC, 1)


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
    significance_test(estimate::EICEstimate, Ψ₀=0)

Performs a TTest
"""
significance_test(estimate::EICEstimate, Ψ₀=0) = OneSampleTTest(estimate, Ψ₀)

Base.show(io::IO, mime::MIME"text/plain", est::Union{EICEstimate, ComposedEstimate}) =
    show(io, mime, significance_test(est))