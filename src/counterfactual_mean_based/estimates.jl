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
    treatments_factor::Union{JointConditionalDistributionEstimate, RieszRepresenterEstimate}
end

string_repr(estimate::MLCMRelevantFactors) = string(
    "Composite Factor Estimate: \n",
    "-------------------------\n- ",
    string_repr(estimate.outcome_mean),"\n- ", 
    string_repr(estimate.treatments_factor)
)

#####################################################################
###                       FoldsMLCMRelevantFactors                     ###
#####################################################################

struct FoldsMLCMRelevantFactors <: Estimate
    estimand::CMRelevantFactors
    estimates::Vector{MLCMRelevantFactors}
end

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
Distributions.estimate(Ψ̂::EICEstimate) = Ψ̂.estimate

Statistics.std(Ψ̂::EICEstimate) = Ψ̂.std

Base.show(io::IO, mime::MIME"text/plain", est::Union{EICEstimate, JointEstimate, ComposedEstimate}) =
    show(io, mime, significance_test(est))