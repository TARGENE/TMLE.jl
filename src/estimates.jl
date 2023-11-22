

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
    std::Matrix{T}
    n::Int
end

function Base.show(io::IO, ::MIME"text/plain", est::ComposedEstimate)
    if length(est.std) !== 1
        println(io, string("Estimate: ", estimate(est), "\nVariance: \n", var(est)))
    else
        testresult = OneSampleTTest(est)
        data = [estimate(est) confint(testresult) pvalue(testresult);]
        headers = ["Estimate", "95% Confidence Interval", "P-value"]
        pretty_table(io, data;header=headers)
    end
end

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
    length(est.std) == 1 ? est.std[1] / est.n : est.std ./ est.n


"""
    OneSampleTTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleTTest(est::ComposedEstimate, Ψ₀=0) 
    @assert length(est.estimate) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleTTest(estimate(est), sqrt(est.std[1]), est.n, Ψ₀)
end

"""
    OneSampleZTest(r::ComposedEstimate, Ψ₀=0)

Performs a T test on the ComposedEstimate.
"""
function HypothesisTests.OneSampleZTest(est::ComposedEstimate, Ψ₀=0) 
    @assert length(est.estimate) == 1 "OneSampleTTest is only implemeted for real-valued statistics."
    return OneSampleZTest(estimate(est), sqrt(est.std[1]), est.n, Ψ₀)
end
