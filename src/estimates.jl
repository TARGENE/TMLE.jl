

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

"""
Prediction for the subset of X identified by idx and fold.
"""
function fold_prediction(estimate::SampleSplitMLConditionalDistribution, X, idx, fold)
    Xval = selectrows(X, idx)
    return predict(estimate.machines[fold], Xval)
end

"""
In the case where newpreds is a UnivariateFiniteVector, we update the probability matrix.
"""
function update_preds!(probs::Matrix, newpreds::UnivariateFiniteVector, idx)
    for (key, vals) in newpreds.prob_given_ref
        probs[idx, key] = vals
    end
end

update_preds!(ŷ, preds, idx) = ŷ[idx] = preds

"""
In the case where predictions are a UnivariateFiniteVector, we store a Matrix of probabilities.
"""
initialize_cv_preds(first_preds::UnivariateFiniteVector, n) = 
    Matrix{Float64}(undef, n, length(first_preds.prob_given_ref))

"""
As a default, we initialize predictions with a Vector of the type corresponding to the
predictions from the first machine.
"""
initialize_cv_preds(first_preds, n) = 
    Vector{eltype(first_preds)}(undef, n)
    
"""
In the case where predictions are a UnivariateFiniteVector, we create a special 
UnivariateFinite vector for downstream optimizaton.
"""
finalize_cv_preds(probs, first_preds::UnivariateFiniteVector) = UnivariateFinite(support(first_preds), probs)

"""
As a default we simply return the vector
"""
finalize_cv_preds(ŷ, first_preds) = ŷ

"""
Out of fold prediction, predictions for fold k are made from machines trained on fold k̄.
We distinguish the case where preidctions are a UnivariateFiniteVector that requires specific attention.
"""
function cv_predict(estimate, X)
    fold_to_val_idx = [(fold, val_idx) for (fold, (_, val_idx)) in enumerate(estimate.train_validation_indices)]
    first_fold, first_val_idx = first(fold_to_val_idx)
    first_preds = fold_prediction(estimate, X, first_val_idx, first_fold)

    ŷ = initialize_cv_preds(first_preds, nrows(X))
    update_preds!(ŷ, first_preds, first_val_idx)
    for (fold, val_idx) in fold_to_val_idx[2:end]
        preds = fold_prediction(estimate, X, val_idx, fold)
        update_preds!(ŷ, preds, val_idx)
    end
    return finalize_cv_preds(ŷ, first_preds)
end

function MLJBase.predict(estimate::SampleSplitMLConditionalDistribution, dataset)
    X = selectcols(dataset, estimate.estimand.parents)
    return cv_predict(estimate, X)
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

function compute_offset(ŷ::AbstractVector{<:UnivariateFinite{<:Union{OrderedFactor{2}, Multiclass{2}}}})
    μy = expected_value(ŷ)
    logit!(μy)
    return μy
end

compute_offset(ŷ::AbstractVector{<:Distributions.UnivariateDistribution}) = expected_value(ŷ)

compute_offset(ŷ::AbstractVector{T}) where T<:Real = expected_value(ŷ)

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

"""
    significance_test(estimate::ComposedEstimate, Ψ₀=zeros(size(estimate.estimate, 1)))

Performs a TTest if the estimate is one dimensional and a HotellingT2Test otherwise.
"""
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