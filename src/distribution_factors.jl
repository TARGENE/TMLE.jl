abstract type DistributionFactor end

function key(f::DistributionFactor) end

mutable struct ConditionalDistribution <: DistributionFactor
    outcome::Symbol
    parents::Set{Symbol}
    model::MLJBase.Supervised
    machine::MLJBase.Machine
    function ConditionalDistribution(outcome, parents, model)
        outcome = Symbol(outcome)
        parents = Set(Symbol(x) for x in parents)
        outcome ∉ parents || throw(SelfReferringEquationError(outcome))
        return new(outcome, parents, model)
    end
end

ConditionalDistribution(;outcome, parents, model) =
    ConditionalDistribution(outcome, parents, model)

key(cd::ConditionalDistribution) = (cd.outcome, cd.parents)

cond_dist_string(outcome, parents) = string(outcome, " | ", join(parents, ", "))

string_repr(cd::ConditionalDistribution) = cond_dist_string(cd.outcome, cd.parents)

fit_message(cd::ConditionalDistribution) = string("Fitting Conditional Distribution Factor: ", string_repr(cd))

update_message(cd::ConditionalDistribution) = string("Reusing or Updating Conditional Distribution Factor: ", string_repr(cd))

Base.show(io::IO, ::MIME"text/plain", cd::ConditionalDistribution) = println(io, string_repr(cd))

function MLJBase.fit!(cd::ConditionalDistribution, dataset; cache=true, verbosity=1, force=false)
    # Never fitted or model has changed
    dofit = !isdefined(cd, :machine) || cd.model != cd.machine.model
    if dofit
        verbosity >= 1 && @info(fit_message(cd))
        featurenames = collect(cd.parents)
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