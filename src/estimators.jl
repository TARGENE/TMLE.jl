abstract type Estimator end

#####################################################################
###               MLConditionalDistributionEstimator              ###
#####################################################################

struct MLConditionalDistributionEstimator <: Estimator
    model::MLJBase.Supervised
end

function (estimator::MLConditionalDistributionEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Lookup in cache
    if haskey(cache, estimand)
        old_estimator, estimate = cache[estimand]
        if key(old_estimator) == key(estimator)
            verbosity > 0 && @info(reuse_string(estimand))
            return estimate
        end
    end
    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    # Otherwise estimate
    relevant_dataset = nomissing(dataset, variables(estimand))
    # Fit Conditional DIstribution using MLJ
    X = selectcols(relevant_dataset, estimand.parents)
    y = Tables.getcolumn(relevant_dataset, estimand.outcome)
    mach = machine(estimator.model, X, y, cache=machine_cache)
    fit!(mach, verbosity=verbosity-1)
    # Build estimate
    estimate = MLConditionalDistribution(estimand, mach)
    # Update cache
    cache[estimand] = estimator => estimate

    return estimate
end

key(estimator::MLConditionalDistributionEstimator) =
    (MLConditionalDistributionEstimator, estimator.model)

#####################################################################
###       SampleSplitMLConditionalDistributionEstimator           ###
#####################################################################

struct SampleSplitMLConditionalDistributionEstimator <: Estimator
    model::MLJBase.Supervised
    train_validation_indices::Tuple
end

function (estimator::SampleSplitMLConditionalDistributionEstimator)(estimand, dataset; cache=Dict(), verbosity=1, machine_cache=false)
    # Lookup in cache
    if haskey(cache, estimand)
        old_estimator, estimate = cache[estimand]
        if key(old_estimator) == key(estimator)
            verbosity > 0 && @info(reuse_string(estimand))
            return estimate
        end
    end
    # Otherwise estimate
    verbosity > 0 && @info(string("Estimating: ", string_repr(estimand)))
    
    relevant_dataset = selectcols(dataset, variables(estimand))
    nfolds = size(estimator.train_validation_indices, 1)
    machines = Vector{Machine}(undef, nfolds)
    # Fit Conditional Distribution on each training split using MLJ
    for (index, (train_indices, _)) in enumerate(estimator.train_validation_indices)
        train_dataset = selectrows(relevant_dataset, train_indices)
        Xtrain = selectcols(train_dataset, estimand.parents)
        ytrain = Tables.getcolumn(train_dataset, estimand.outcome)
        mach = machine(estimator.model, Xtrain, ytrain, cache=machine_cache)
        fit!(mach, verbosity=verbosity-1)
        machines[index] = mach
    end
    # Build estimate
    estimate = SampleSplitMLConditionalDistribution(estimand, estimator.train_validation_indices, machines)
    # Update cache
    cache[estimand] = estimator => estimate

    return estimate
end

key(estimator::SampleSplitMLConditionalDistributionEstimator) =
    (MLConditionalDistributionEstimator, estimator.model, estimator.train_validation_indices)

ConditionalDistributionEstimator(train_validation_indices::Nothing, model) =
    MLConditionalDistributionEstimator(model)

ConditionalDistributionEstimator(train_validation_indices, model) =
    SampleSplitMLConditionalDistributionEstimator(model, train_validation_indices)

#####################################################################
###                   ComposedEstimand Estimator                  ###
#####################################################################

"""
    (estimator::Estimator)(Ψ::ComposedEstimand, dataset; cache=Dict(), verbosity=1)

Estimates all components of Ψ and then Ψ itself.
"""
function (estimator::Estimator)(Ψ::ComposedEstimand, dataset; cache=Dict(), verbosity=1, backend=DI.AutoZygote())
    estimates = map(Ψ.args) do estimand 
        estimate, _ = estimator(estimand, dataset; cache=cache, verbosity=verbosity)
        estimate
    end
    f₀, σ₀, n = _compose(Ψ.f, estimates...; backend=backend)
    return ComposedEstimate(Ψ, estimates, f₀, σ₀, n), cache
end

"""
    joint_estimand(args...)
    
This function is temporary and only necessary because of a bug in 
the AD package. Simply call `vcat` in the future.
"""
joint_estimand(args...) = vcat(args...)

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
function compose(f, estimates...; backend=DI.AutoZygote())
    f₀, σ₀, n = _compose(f, estimates...; backend=backend)
    estimand = ComposedEstimand(f, Tuple(e.estimand for e in estimates))
    return ComposedEstimate(estimand, estimates, f₀, σ₀, n)
end

_make_vec(x::Number) = [x]
_make_vec(x::AbstractVector) = x

function _compose(f, estimates...; backend=DI.AutoZygote())
    Σ = covariance_matrix(estimates...)
    point_estimates = [r.estimate for r in estimates]
    f₀, J = DI.value_and_jacobian(_make_vec ∘ Base.splat(f), backend, point_estimates)
    n = size(first(estimates).IC, 1)
    σ₀ = J * Σ * J'
    return f₀, σ₀, n
end

function covariance_matrix(estimates...)
    X = hcat([r.IC for r in estimates]...)
    return cov(X, dims=1, corrected=true)
end
