"""
This structure is used as a cache for both:
    - η_spec: The specification of the learning algorithms used to estimate the nuisance parameters
    - η: nuisance parameters
    - data: dataset changes

If the input data does not change, then the nuisance parameters do not have to be estimated again.
"""
mutable struct TMLECache
    Ψ::Parameter
    η_spec::NuisanceSpec
    dataset
    η::NuisanceParameters
    mach_cache::Bool
    function TMLECache(Ψ, η_spec, dataset, mach_cache)
        no_missing_dataset = nomissing(dataset, allcolumns(Ψ))
        dataset = Dict(
            :source => dataset,
            :no_missing => no_missing_dataset,
            :joint_treatment => joint_treatment(treatments(no_missing_dataset, Ψ))
        )
        η = NuisanceParameters(nothing, nothing, nothing, nothing)
        new(Ψ, η_spec, dataset, η, mach_cache)
    end
end

function update!(cache::TMLECache, Ψ::Parameter)
    any_variable_changed = false
    treatment_changed = false
    if keys(cache.Ψ.treatment) != keys(Ψ.treatment)
        cache.η.G = nothing
        cache.η.Q = nothing
        cache.η.H = nothing
        any_variable_changed = true
        treatment_changed = true
    end
    if cache.Ψ.confounders != Ψ.confounders
        cache.η.G = nothing
        cache.η.Q = nothing
        any_variable_changed = true
    end
    if cache.Ψ.covariates != Ψ.covariates
        cache.η.Q = nothing
        any_variable_changed = true
    end
    if cache.Ψ.target != Ψ.target
        cache.η.Q = nothing
        any_variable_changed = true
    end
    cache.η.F = nothing
    # Update no missing dataset
    if any_variable_changed
        cache.dataset[:no_missing] = nomissing(cache.dataset[:source], allcolumns(Ψ))
    end
    cache.Ψ = Ψ
end

function update!(cache::TMLECache, η_spec::NuisanceSpec)
    if cache.η_spec.G != η_spec.G
        cache.η.G = nothing
    end
    if cache.η_spec.Q != η_spec.Q
        cache.η.Q = nothing
    end
    cache.η.F = nothing
    cache.η_spec = η_spec
end

function update!(cache::TMLECache, Ψ::Parameter, η_spec::NuisanceSpec)
    update!(cache, Ψ)
    update!(cache, η_spec)
end

"""
    tmle(Ψ::Parameter, η_spec::NuisanceSpec, dataset; verbosity=1, threshold=1e-8)

Main entrypoint to run the TMLE procedure.

# Arguments

- Ψ: The parameter of interest
- η_spec: The specification for learning `Q_0` and `G_0`
- dataset: A tabular dataset respecting the Table.jl interface
- verbosity: The logging level
- threshold: To avoid small values of Ĝ to cause the "clever covariate" to explode
- mach_cache: Whether underlying MLJ.machines will cache data or not
"""
function tmle(Ψ::Parameter, η_spec::NuisanceSpec, dataset; verbosity=1, threshold=1e-8, mach_cache=false)
    cache = TMLECache(Ψ, η_spec, dataset, mach_cache)
    return tmle!(cache; verbosity=verbosity, threshold=threshold)
end

function tmle!(cache::TMLECache; verbosity=1, threshold=1e-8)
    # Initial fit of the nuisance parameters
    verbosity >= 1 && @info "Fitting the nuisance parameters..."
    TMLE.fit_nuisance!(cache, verbosity=verbosity)
    
    # TMLE step
    verbosity >= 1 && @info "Targeting the nuisance parameters..."
    tmle_step!(cache, verbosity=verbosity, threshold=threshold)
    
    # Estimation results after TMLE
    IC, Ψ̂, Ψ̂ᵢ = TMLE.gradient_and_estimates(cache)
    result = PointTMLE(Ψ̂, IC, Ψ̂ᵢ)

    verbosity >= 1 && @info "Done."
    return result, cache
end

"""
    tmle!(cache::TMLECache, Ψ::Parameter; verbosity=1, threshold=1e-8)

Runs the TMLE procedure for the new parameter Ψ while potentially reusing cached nuisance parameters.
"""
function tmle!(cache::TMLECache, Ψ::Parameter; verbosity=1, threshold=1e-8)
    update!(cache, Ψ)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end

"""
    tmle!(cache::TMLECache, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8)

Runs the TMLE procedure for the new nuisance parameters specification η_spec while potentially reusing cached nuisance parameters.
"""
function tmle!(cache::TMLECache, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8, mach_cache=false)
    update!(cache, η_spec)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end

"""
    tmle!(cache::TMLECache, Ψ::Parameter, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8)

Runs the TMLE procedure for the new parameter Ψ and the new nuisance parameters specification η_spec 
while potentially reusing cached nuisance parameters.
"""
function tmle!(cache::TMLECache, Ψ::Parameter, η_spec::NuisanceSpec; verbosity=1, threshold=1e-8)
    update!(cache, Ψ, η_spec)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end

"""
    tmle!(cache::TMLECache, η_spec::NuisanceSpec, Ψ::Parameter; verbosity=1, threshold=1e-8)

Runs the TMLE procedure for the new parameter Ψ and the new nuisance parameters specification η_spec 
while potentially reusing cached nuisance parameters.
"""
function tmle!(cache::TMLECache, η_spec::NuisanceSpec, Ψ::Parameter; verbosity=1, threshold=1e-8)
    update!(cache, Ψ, η_spec)
    tmle!(cache, verbosity=verbosity, threshold=threshold)
end


"""
    fit_nuisance!(cache::TMLECache; verbosity=1, mach_cache=false)
    
Fits the nuisance parameters η on the dataset using the specifications from η_spec
and the variables defined by Ψ.
"""
function fit_nuisance!(cache::TMLECache; verbosity=1)
    Ψ, η_spec, η = cache.Ψ, cache.η_spec, cache.η
    # Fitting P(T|W)
    # Only rows with missing values in either W or Tₜ are removed
    if η.G === nothing
        log_fit(verbosity, "P(T|W)")
        nomissing_WT = nomissing(cache.dataset[:source], treatment_and_confounders(Ψ))
        W = confounders(nomissing_WT, Ψ)
        jointT = joint_treatment(treatments(nomissing_WT, Ψ))
        mach = machine(η_spec.G, W, jointT, cache=cache.mach_cache)
        MLJBase.fit!(mach, verbosity=verbosity-1)
        η.G = mach
        cache.dataset[:jointT_levels] = levels(jointT)
    else
        log_no_fit(verbosity, "P(T|W)")
    end

    # Fitting E[Y|X]
    if η.Q === nothing
        # Data
        X = Qinputs(cache.dataset[:no_missing], Ψ)
        y = target(cache.dataset[:no_missing], Ψ)

        # Fitting the Encoder
        if η.H === nothing
            log_fit(verbosity, "Encoder")
            mach = machine(η_spec.H, X, cache=cache.mach_cache)
            MLJBase.fit!(mach, verbosity=verbosity-1)
            η.H = mach
        else
            log_no_fit(verbosity, "Encoder")
        end
        log_fit(verbosity, "E[Y|X]")
        Xfloat = MLJBase.transform(η.H, X)
        cache.dataset[:Xfloat] = Xfloat
        mach = machine(η_spec.Q, Xfloat, y, cache=cache.mach_cache)
        MLJBase.fit!(mach, verbosity=verbosity-1)
        η.Q = mach
    else
        log_no_fit(verbosity, "Encoder")
        log_no_fit(verbosity, "E[Y|X]")
    end
end


function tmle_step!(cache::TMLECache; verbosity=1, threshold=1e-8)
    # Fit fluctuation
    ŷ = MLJBase.predict(cache.η.Q, cache.dataset[:Xfloat])
    offset = TMLE.compute_offset(ŷ)
    W = TMLE.confounders(cache.dataset[:no_missing], cache.Ψ)
    jointT = TMLE.joint_treatment(TMLE.treatments(cache.dataset[:no_missing], cache.Ψ))
    covariate = TMLE.compute_covariate(jointT, W, cache.Ψ, cache.η.G; threshold=threshold)
    X = TMLE.fluctuation_input(covariate, offset)
    y = TMLE.target(cache.dataset[:no_missing], cache.Ψ)
    mach = machine(cache.η_spec.F, X, y, cache=cache.mach_cache)
    MLJBase.fit!(mach, verbosity=verbosity-1)
    # Update cache
    cache.η.F = mach
    # This is useful for gradient_Y_X
    cache.dataset[:covariate] = X.covariate
    cache.dataset[:μy] = TMLE.expected_value(MLJBase.predict(mach, X))
end

function counterfactual_aggregates(cache::TMLECache; threshold=1e-8)
    dataset = cache.dataset[:no_missing]
    WC = TMLE.confounders_and_covariates(dataset, cache.Ψ)
    Ttemplate = TMLE.treatments(dataset, cache.Ψ)
    n = nrows(Ttemplate)
    counterfactual_aggregateᵢ = zeros(n)
    counterfactual_aggregate = zeros(n)
    # Loop over Treatment settings
    for (vals, sign) in TMLE.indicator_fns(cache.Ψ, x -> x)
        # Counterfactual dataset for a given treatment setting
        Tc = TMLE.counterfactualTreatment(vals, Ttemplate)
        Xc = Qinputs(merge(WC, Tc), cache.Ψ)
        # Counterfactual predictions with the initial Q
        ŷᵢ = MLJBase.predict(cache.η.Q,  MLJBase.transform(cache.η.H, Xc))
        counterfactual_aggregateᵢ .+= sign .* expected_value(ŷᵢ)
        # Counterfactual predictions with F
        offset = compute_offset(ŷᵢ)
        jointT = categorical(repeat([joint_name(vals)], n), levels=cache.dataset[:jointT_levels])
        covariate = compute_covariate(jointT, confounders(WC, cache.Ψ), cache.Ψ, cache.η.G; threshold=threshold)
        Xfluct = fluctuation_input(covariate, offset)
        ŷ = predict(cache.η.F, Xfluct)
        counterfactual_aggregate .+= sign .* expected_value(ŷ)
    end
    return counterfactual_aggregate, counterfactual_aggregateᵢ
end

"""
    gradient_W(counterfactual_aggregate, estimate)

∇_W = counterfactual_aggregate - Ψ
"""
gradient_W(counterfactual_aggregate, estimate) =
    counterfactual_aggregate .- estimate


"""
    gradient_Y_X(cache)

∇_YX(w, t, c) = covariate(w, t)  ̇ (y - E[Y|w, t, c])

This part of the gradient is evaluated on the original dataset. All quantities have been precomputed and cached.
"""
function gradient_Y_X(cache::TMLECache)
    covariate = cache.dataset[:covariate]
    y = target(cache.dataset[:no_missing], cache.Ψ)
    return covariate .* (float(y) .- cache.dataset[:μy])
end


function gradient_and_estimates(cache::TMLECache; threshold=1e-8)
    counterfactual_aggregate, counterfactual_aggregateᵢ = TMLE.counterfactual_aggregates(cache; threshold=threshold)
    Ψ̂, Ψ̂ᵢ = mean(counterfactual_aggregate), mean(counterfactual_aggregateᵢ)
    IC = gradient_Y_X(cache) .+ gradient_W(counterfactual_aggregate, Ψ̂)
    return IC, Ψ̂, Ψ̂ᵢ
end

