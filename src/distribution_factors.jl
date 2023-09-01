abstract type DistributionFactor end

function key(f::DistributionFactor) end

mutable struct ConditionalDistribution{T <: MLJBase.Supervised} <: DistributionFactor
    outcome::Symbol
    parents::Set{Symbol}
    model::T
    machine::MLJBase.Machine
    function ConditionalDistribution{T}(outcome, parents, model::T) where T <: MLJBase.Supervised
        outcome = Symbol(outcome)
        parents = Set(Symbol(x) for x in parents)
        outcome ∉ parents || throw(SelfReferringEquationError(outcome))
        return new(outcome, parents, model)
    end
end

ConditionalDistribution(outcome, parents, model::T) where T <: MLJBase.Supervised = 
    ConditionalDistribution{T}(outcome, parents, model)

ConditionalDistribution(;outcome, parents, model) =
    ConditionalDistribution(outcome, parents, model)

key(cd::ConditionalDistribution) = (cd.outcome, cd.parents)

cond_dist_string(outcome, parents) = string(outcome, " | ", join(parents, ", "))

string_repr(cd::ConditionalDistribution) = cond_dist_string(cd.outcome, cd.parents)

fit_message(cd::ConditionalDistribution) = string("Fitting Conditional Distribution Factor: ", string_repr(cd))

update_message(cd::ConditionalDistribution) = string("Reusing or Updating Conditional Distribution Factor: ", string_repr(cd))

Base.show(io::IO, ::MIME"text/plain", cd::ConditionalDistribution) = println(io, string_repr(cd))

get_featurenames(cd::ConditionalDistribution) = sort(collect(cd.parents))

function MLJBase.fit!(cd::ConditionalDistribution, dataset; cache=true, verbosity=1, force=false)
    # Never fitted or model has changed
    dofit = !isdefined(cd, :machine) || cd.model != cd.machine.model
    if dofit
        verbosity >= 1 && @info(fit_message(cd))
        featurenames = get_featurenames(cd)
        data = nomissing(dataset, vcat(featurenames, cd.outcome))
        X = selectcols(data, featurenames)
        y = Tables.getcolumn(data, cd.outcome)
        mach = machine(cd.model, X, y, cache=cache)
        MLJBase.fit!(mach, verbosity=verbosity-1)
        cd.machine = mach
    # Also refit if force is true
    else
        verbosity >= 1 && @info(update_message(cd))
        MLJBase.fit!(cd.machine, verbosity=verbosity-1, force=force)
    end
end

MLJBase.fitted_params(cd::ConditionalDistribution) = fitted_params(cd.machine)

function MLJBase.predict(cd::ConditionalDistribution, dataset)
    featurenames = get_featurenames(cd)
    X = selectcols(dataset, featurenames)
    return predict(cd.machine, X)
end

function expected_value(factor::ConditionalDistribution, dataset)
    return expected_value(predict(factor, dataset))
end

function likelihood(factor::ConditionalDistribution, dataset)
    ŷ = predict(factor, dataset)
    y = Tables.getcolumn(dataset, factor.outcome)
    return pdf.(ŷ, y)
end

## CVCounterPart

mutable struct CVConditionalDistribution
    outcome::Symbol
    parents::Set{Symbol}
    model::MLJBase.Supervised
    resampling::ResamplingStrategy
    machines::Vector{MLJBase.Machine}
end

function MLJBase.fit!(factor::CVConditionalDistribution, dataset; cache=true, verbosity=1, force=false)
    verbosity >= 1 && @info(fit_message(factor))
    featurenames = get_featurenames(cd)
    data = nomissing(dataset, vcat(featurenames, cd.outcome))
    X = selectcols(data, featurenames)
    y = Tables.getcolumn(data, cd.outcome)
    machines = Machine[]
    for (train, val) in MLJBase.train_test_pairs(resampling, nrows(data), data)
        Xtrain = selectrows(X, train)
        ytrain = selectrows(y, train)
        mach = machine(factor.model, Xtrain, ytrain)
        fit!(mach, verbosity=verbosity-1)
        push!(factor.machines, mach)
    end
    factor.machines = machines
end

function MLJBase.predict(factor::CVConditionalDistribution, dataset)
    ŷ = Vector{Any}(undef, size(dataset, 1))
    for (fold, (_, val)) in enumerate(MLJBase.train_test_pairs(resampling, nrows(data), data))
        Xval = selectrows(X, val)
        ŷ[val] = predict(factor.machines[fold], Xval)
    end
    return ŷ
end
