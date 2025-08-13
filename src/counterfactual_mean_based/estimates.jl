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
    propensity_score
end

string_repr(estimate::MLCMRelevantFactors) = string(
    "Composite Factor Estimate: \n",
    "-------------------------\n- ",
    string_repr(estimate.outcome_mean),"\n- ", 
    join((string_repr(f) for f in estimate.propensity_score.components), "\n- ")
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

function print_header(io::IO, est::JointEstimate{T, E, N}) where {T, E <: TMLEstimate, N}
    println(io, "Joint Targeted Minimum Loss Based Estimator")
    println(io, "-------------------------------------------")
end

struct OSEstimate{T<:AbstractFloat} <: Estimate
    estimand::StatisticalCMCompositeEstimand
    estimate::T
    std::T
    n::Int
    IC::Vector{T}
end

OSEstimate(;estimand, estimate::T, std::T, n, IC) where T = OSEstimate(estimand, estimate, std, n, convert(Vector{T}, IC))

function print_header(io::IO, est::JointEstimate{T, E, N}) where {T, E <: OSEstimate, N}
    println(io, "Joint One Step Estimator")
    println(io, "------------------------")
end

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


function print_header(io::IO, Ψ̂::TMLEstimate)
    println(io, "Targeted Minimum Loss Based Estimator")
    println(io, "-------------------------------------")
end

function print_header(io::IO, Ψ̂::OSEstimate)
    println(io, "One Step Estimator")
    println(io, "------------------")
end

function Base.show(io::IO, mime::MIME"text/plain", est::EICEstimate)
    test_result = significance_test(est)
    print_header(io, est)
    println(io, "- point estimate         : ", @sprintf("%.4f", est.estimate))
    println(io, "- 95% confidence interval: ", @sprintf("[%.4f, %.4f]", confint(test_result)...))
    println(io, "- p-value                : ", pretty_pvalue(pvalue(test_result)))
    println(io, "- mean influence curve   : ", @sprintf("%.2e", mean(est.IC)))
    println(io, "\nFull test results can be obtained with `significance_test`")
end