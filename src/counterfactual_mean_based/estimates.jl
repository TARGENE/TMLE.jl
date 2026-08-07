#####################################################################
###                       MLCMRelevantFactors                     ###
#####################################################################

"""
Holds a Sample Split Machine Learning set of estimates (outcome mean, propensity score) 
for counterfactual mean based estimands' relevant factors.

"""
struct MLCMRelevantFactors{C <: Union{Nothing, <:ConditionalDistributionEstimate}} <: Estimate
    estimand::CMRelevantFactors
    outcome_mean::ConditionalDistributionEstimate
    propensity_score
    censoring_score::C
end

MLCMRelevantFactors(estimand, outcome_mean, propensity_score) =
    MLCMRelevantFactors(estimand, outcome_mean, propensity_score, nothing)

getG(factors::MLCMRelevantFactors{Nothing}) = factors.propensity_score

getG(factors::MLCMRelevantFactors{<:ConditionalDistributionEstimate}) = (factors.propensity_score, factors.censoring_score)

"""
    align_to_rows(factors::MLCMRelevantFactors, keep::Vector{Int})

Realign every nuisance's CV fold indices from the full initial-dataset row space to the
covariate-complete fluctuation subset (see `align_to_rows` on the component estimates).
"""
align_to_rows(factors::MLCMRelevantFactors, keep) = MLCMRelevantFactors(
    factors.estimand,
    align_to_rows(factors.outcome_mean, keep),
    align_to_rows(factors.propensity_score, keep),
    factors.censoring_score === nothing ? nothing : align_to_rows(factors.censoring_score, keep)
)

function string_repr(estimate::MLCMRelevantFactors)
    parts = [
        "Composite Factor Estimate: \n",
        "-------------------------\n- ",
        string_repr(estimate.outcome_mean), "\n- ",
        join((string_repr(f) for f in estimate.propensity_score.components), "\n- ")
    ]
    if estimate.censoring_score !== nothing
        push!(parts, "\n- ", string_repr(estimate.censoring_score))
    end
    return string(parts...)
end

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