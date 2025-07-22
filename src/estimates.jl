

abstract type Estimate end

string_repr(estimate::Estimate) = estimate

Base.show(io::IO, ::MIME"text/plain", estimate::Estimate) =
    println(io, string_repr(estimate))

#####################################################################
###                         MLJEstimate                           ###
#####################################################################

"""
Holds a Machine Learning estimate for a Conditional Distribution.
"""
struct MLJEstimate{T} <: Estimate
    estimand::T
    machine::MLJBase.Machine
end

string_repr(estimate::MLJEstimate{ConditionalDistribution}) = string(
    "P̂(", estimate.estimand.outcome, " | ", join(estimate.estimand.parents, ", "), 
    "), with model: ", 
    Base.typename(typeof(estimate.machine.model)).wrapper
)

string_repr(estimate::MLJEstimate{<:RieszRepresenter}) = string(
    "α̂(", join(variables(estimate.estimand), ", "), 
    "), with model: ", 
    Base.typename(typeof(estimate.machine.model)).wrapper
)

function MLJBase.predict(estimate::MLJEstimate, dataset)
    X = get_mlj_inputs(estimate.estimand, dataset)
    return MLJBase.predict(estimate.machine, X)
end

#####################################################################
###                       SampleSplitMLJEstimate                  ###
#####################################################################

"""
Holds a Sample Split Machine Learning estimate for a Conditional Distribution.
Each machine in `machines` contains a ML model trained on a different training fold of the data.
The predictions are made out of fold, i.e. for each fold k, the predictions are made using the machine trained on fold k̄.
"""
struct SampleSplitMLJEstimate{T} <: Estimate
    estimand::T
    train_validation_indices
    machines::Vector{Machine}
end

string_repr(estimate::SampleSplitMLJEstimate{ConditionalDistribution}) = 
    string("P̂(", estimate.estimand.outcome, " | ", join(estimate.estimand.parents, ", "), 
    "), sample split with model: ", 
    Base.typename(typeof(first(estimate.machines).model)).wrapper
)

string_repr(estimate::SampleSplitMLJEstimate{<:RieszRepresenter}) = 
    string("α̂(", join(variables(estimate.estimand), ", "), 
    "), sample split with model: ", 
    Base.typename(typeof(first(estimate.machines).model)).wrapper
)

"""
Prediction for the subset of X identified by idx and fold.
"""
function fold_prediction(estimate::SampleSplitMLJEstimate, X, idx, fold)
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

function MLJBase.predict(estimate::SampleSplitMLJEstimate, dataset)
    X = get_mlj_inputs(estimate.estimand, dataset)
    return cv_predict(estimate, X)
end

#####################################################################
###               ConditionalDistributionEstimate                 ###
#####################################################################

ConditionalDistributionEstimate = Union{MLJEstimate{ConditionalDistribution}, SampleSplitMLJEstimate{ConditionalDistribution}}

function expected_value(estimate::ConditionalDistributionEstimate, dataset)
    return expected_value(predict(estimate, dataset))
end

function likelihood(estimate::ConditionalDistributionEstimate, dataset)
    ŷ = predict(estimate, dataset)
    y = dataset[!, estimate.estimand.outcome]
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
###            JointConditionalDistributionEstimate               ###
#####################################################################

struct JointConditionalDistributionEstimate{T, N} <: Estimate
    estimand::JointConditionalDistribution{N}
    components::Tuple{Vararg{T, N}}
end

Base.getindex(estimand::JointConditionalDistributionEstimate, i) = Base.getindex(estimand.components, i)

Base.firstindex(estimand::JointConditionalDistributionEstimate) = Base.firstindex(estimand.components)

Base.lastindex(estimand::JointConditionalDistributionEstimate) = Base.lastindex(estimand.components)

Base.iterate(estimand::JointConditionalDistributionEstimate) = Base.iterate(estimand.components)

Base.iterate(estimand::JointConditionalDistributionEstimate, state) = Base.iterate(estimand.components, state)

Base.length(estimand::JointConditionalDistributionEstimate) = length(estimand.components)

Base.IteratorSize(::Type{JointConditionalDistributionEstimate}) = Base.HasLength()

Base.IteratorEltype(::Type{JointConditionalDistributionEstimate}) = Base.HasEltype()

Base.eltype(::Type{JointConditionalDistributionEstimate{T, N}}) where {T, N} = T

#####################################################################
###                        Joint Estimate                         ###
#####################################################################

struct JointEstimate{T<:AbstractFloat} <: Estimate
    estimand::JointEstimand
    estimates::Tuple
    cov::Matrix{T}
    n::Int
end

to_matrix(x::Matrix) = x
to_matrix(x) = reduce(hcat, x)

JointEstimate(;estimand, estimates, cov, n) =
    JointEstimate(estimand, Tuple(estimates), to_matrix(cov), n)

"""
    Distributions.estimate(r::JointEstimate)

Retrieves the final estimate: after the TMLE step.
"""
Distributions.estimate(Ψ̂::JointEstimate) = [x.estimate for x in Ψ̂.estimates]

Statistics.std(Ψ̂::JointEstimate) = sqrt(only(Ψ̂.cov))

function emptyIC(estimate::JointEstimate, pval_threshold)
    emptied_estimates = Tuple(emptyIC(e, pval_threshold) for e in estimate.estimates)
    JointEstimate(estimate.estimand, emptied_estimates, estimate.cov, estimate.n)
end

to_dict(estimate::JointEstimate) = Dict(
    :type => string(JointEstimate),
    :estimand => to_dict(estimate.estimand),
    :estimates => [to_dict(e) for e in estimate.estimates],
    :cov => estimate.cov,
    :n => estimate.n
)


#####################################################################
###                       Composed Estimate                       ###
#####################################################################

struct ComposedEstimate{T<:AbstractFloat} <: Estimate
    estimand::ComposedEstimand
    estimates::Vector{T}
    cov::Matrix{T}
    n::Int
end

ComposedEstimate(Ψ, estimate::Real, cov, n) = ComposedEstimate(Ψ, [estimate], cov, n)

ComposedEstimate(;estimand, estimates, cov, n) = ComposedEstimate(estimand, estimates, cov, n)

Distributions.estimate(Ψ̂::ComposedEstimate) = Ψ̂.estimates

Statistics.std(Ψ̂::ComposedEstimate) = sqrt(only(Ψ̂.cov))

function to_dict(Ψ̂::ComposedEstimate)
    Dict(
    :type => string(ComposedEstimate),
    :estimand => to_dict(Ψ̂.estimand),
    :estimates => Ψ̂.estimates,
    :cov => Ψ̂.cov,
    :n => Ψ̂.n
)
end