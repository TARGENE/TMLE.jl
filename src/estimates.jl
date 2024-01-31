

abstract type Estimate end

string_repr(estimate::Estimate) = estimate

Base.show(io::IO, ::MIME"text/plain", estimate::Estimate) =
    println(io, string_repr(estimate))

#####################################################################
###                   MLConditionalDistribution                   ###
#####################################################################

"""
Holds a Machine Learning estimate for a Conditional Distribution.
"""
struct MLConditionalDistribution <: Estimate
    estimand::ConditionalDistribution
    machine::MLJBase.Machine
end

string_repr(estimate::MLConditionalDistribution) = string(
    "P̂(", estimate.estimand.outcome, " | ", join(estimate.estimand.parents, ", "), 
    "), with model: ", 
    Base.typename(typeof(estimate.machine.model)).wrapper
)

function MLJBase.predict(estimate::MLConditionalDistribution, dataset)
    X = selectcols(dataset, estimate.estimand.parents)
    return predict(estimate.machine, X)
end


#####################################################################
###             SampleSplitMLConditionalDistribution              ###
#####################################################################

"""
Holds a Sample Split Machine Learning estimate for a Conditional Distribution.
"""
struct SampleSplitMLConditionalDistribution <: Estimate
    estimand::ConditionalDistribution
    train_validation_indices::Tuple
    machines::Vector{Machine}
end

string_repr(estimate::SampleSplitMLConditionalDistribution) = 
    string("P̂(", estimate.estimand.outcome, " | ", join(estimate.estimand.parents, ", "), 
    "), sample split with model: ", 
    Base.typename(typeof(first(estimate.machines).model)).wrapper
)

function MLJBase.predict(estimate::SampleSplitMLConditionalDistribution, dataset)
    X = selectcols(dataset, estimate.estimand.parents)
    ŷs = []
    for (fold, (_, validation_indices)) in enumerate(estimate.train_validation_indices)
        Xval = selectrows(X, validation_indices)
        push!(ŷs, predict(estimate.machines[fold], Xval))
    end
    return vcat(ŷs...)
end

#####################################################################
###               ConditionalDistributionEstimate                 ###
#####################################################################

ConditionalDistributionEstimate = Union{MLConditionalDistribution, SampleSplitMLConditionalDistribution}

function expected_value(estimate::ConditionalDistributionEstimate, dataset)
    return expected_value(predict(estimate, dataset))
end

function likelihood(estimate::ConditionalDistributionEstimate, dataset)
    ŷ = predict(estimate, dataset)
    y = Tables.getcolumn(dataset, estimate.estimand.outcome)
    return pdf.(ŷ, y)
end

function compute_offset(ŷ::UnivariateFiniteVector{Multiclass{2}})
    μy = expected_value(ŷ)
    logit!(μy)
    return μy
end

compute_offset(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = expected_value(ŷ)

compute_offset(ŷ::AbstractVector{<:Real}) = expected_value(ŷ)

function compute_offset(estimate::ConditionalDistributionEstimate, X)
    ŷ = predict(estimate, X)
    return compute_offset(ŷ)
end

#####################################################################
###                       Composed Estimate                      ###
#####################################################################

struct ComposedEstimate{T<:AbstractFloat} <: Estimate
    estimand::ComposedEstimand
    estimates::Tuple
    estimate::Array{T}
    cov::Matrix{T}
    n::Int
end

to_matrix(x::Matrix) = x
to_matrix(x) = reduce(hcat, x)

ComposedEstimate(;estimand, estimates, estimate, cov, n) =
    ComposedEstimate(estimand, Tuple(estimates), collect(estimate), to_matrix(cov), n)

"""
    Distributions.estimate(r::ComposedEstimate)

Retrieves the final estimate: after the TMLE step.
"""
Distributions.estimate(est::ComposedEstimate) = 
    length(est.estimate) == 1 ? est.estimate[1] : est.estimate


"""
    var(r::ComposedEstimate)

Computes the estimated variance associated with the estimate.
"""
Statistics.var(est::ComposedEstimate) = 
    length(est.cov) == 1 ? est.cov[1] / est.n : est.cov ./ est.n

"""
    OneSampleTTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleTTest(estimate::ComposedEstimate, Ψ₀=0) 
    @assert length(estimate.estimate) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleTTest(estimate.estimate[1], sqrt(estimate.cov[1]), estimate.n, Ψ₀)
end

function HypothesisTests.OneSampleHotellingT2Test(estimate::ComposedEstimate, Ψ₀=zeros(size(estimate.estimate, 1)))
    x̄ = estimate.estimate
    S = estimate.cov
    n, p = estimate.n, length(x̄)
    p == length(Ψ₀) ||
        throw(DimensionMismatch("Number of variables does not match number of means"))
    n > 0 || throw(ArgumentError("The input must be non-empty"))
    
    T² = n * HypothesisTests.At_Binv_A(x̄ .- Ψ₀, S)
    F = (n - p) * T² / (p * (n - 1))
    return OneSampleHotellingT2Test(T², F, n, p, Ψ₀, x̄, S)
end

"""
    OneSampleZTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleZTest(estimate::ComposedEstimate, Ψ₀=0) 
    @assert length(estimate.estimate) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleZTest(estimate.estimate[1], sqrt(estimate.cov[1]), estimate.n, Ψ₀)
end

function significance_test(estimate::ComposedEstimate, Ψ₀=zeros(size(estimate.estimate, 1)))
    if length(estimate.estimate) == 1
        Ψ₀ = Ψ₀ isa AbstractArray ? first(Ψ₀) : Ψ₀
        return OneSampleTTest(estimate, Ψ₀)
    else
        return OneSampleHotellingT2Test(estimate, Ψ₀)
    end
end

function emptyIC(estimate::ComposedEstimate, pval_threshold)
    emptied_estimates = Tuple(emptyIC(e, pval_threshold) for e in estimate.estimates)
    ComposedEstimate(estimate.estimand, emptied_estimates, estimate.estimate, estimate.cov, estimate.n)
end


to_dict(estimate::ComposedEstimate) = Dict(
    :type => string(ComposedEstimate),
    :estimand => to_dict(estimate.estimand),
    :estimates => [to_dict(e) for e in estimate.estimates],
    :estimate => estimate.estimate,
    :cov => estimate.cov,
    :n => estimate.n
)