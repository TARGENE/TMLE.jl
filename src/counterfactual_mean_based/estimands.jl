#####################################################################
###                       CMRelevantFactors                       ###
#####################################################################

"""
Defines relevant factors that need to be estimated in order to estimate any
Counterfactual Mean composite estimand (see `StatisticalCMCompositeEstimand`).
"""
struct CMRelevantFactors <: Estimand
    outcome_mean::ConditionalDistribution
    propensity_score::Tuple{Vararg{ConditionalDistribution}}
end

CMRelevantFactors(outcome_mean, propensity_score::ConditionalDistribution) = 
    CMRelevantFactors(outcome_mean, (propensity_score,))

CMRelevantFactors(;outcome_mean, propensity_score) = 
    CMRelevantFactors(outcome_mean, propensity_score)

string_repr(estimand::CMRelevantFactors) = 
    string("Composite Factor: \n",
           "----------------\n- ",
        string_repr(estimand.outcome_mean),"\n- ", 
        join((string_repr(f) for f in estimand.propensity_score), "\n- "))

variables(estimand::CMRelevantFactors) = 
    Tuple(union(variables(estimand.outcome_mean), (variables(est) for est in estimand.propensity_score)...))

#####################################################################
###                         Functionals                           ###
#####################################################################

const ESTIMANDS_DOCS = Dict(
    :CM => (formula="``CM(Y, T=t) = E[Y|do(T=t)]``",),
    :ATE => (formula="``ATE(Y, T, case, control) = E[Y|do(T=case)] - E[Y|do(T=control)``",),
    :IATE => (formula="``IATE = E[Y|do(T₁=1, T₂=1)] - E[Y|do(T₁=1, T₂=0)] - E[Y|do(T₁=0, T₂=1)] + E[Y|do(T₁=0, T₂=0)]``",)
)

for (estimand, (formula,)) ∈ ESTIMANDS_DOCS
    causal_estimand = Symbol(:Causal, estimand)
    statistical_estimand = Symbol(:Statistical, estimand)
    ex = quote
        # Causal Estimand
        struct $(causal_estimand) <: Estimand
            outcome::Symbol
            treatment_values::NamedTuple

            function $(causal_estimand)(outcome, treatment_values)
                outcome = Symbol(outcome)
                treatment_values = get_treatment_specs(treatment_values)
                return new(outcome, treatment_values)
            end
        end
        # Statistical Estimand
        struct $(statistical_estimand) <: Estimand
            outcome::Symbol
            treatment_values::NamedTuple
            treatment_confounders::NamedTuple
            outcome_extra_covariates::Tuple{Vararg{Symbol}}

            function $(statistical_estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
                outcome = Symbol(outcome)
                treatment_values = get_treatment_specs(treatment_values)
                treatment_variables = Tuple(keys(treatment_values))
                treatment_confounders = NamedTuple{treatment_variables}([confounders_values(treatment_confounders, T) for T ∈ treatment_variables])
                outcome_extra_covariates = unique_sorted_tuple(outcome_extra_covariates)
                return new(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
            end
        end

        # Constructors
        $(causal_estimand)(;outcome, treatment_values) =  $(causal_estimand)(outcome, treatment_values)

        $(statistical_estimand)(;outcome, treatment_values, treatment_confounders, outcome_extra_covariates) =
            $(statistical_estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)
        
        # Short Name Constructors
        $(estimand)(outcome, treatment_values) = $(causal_estimand)(outcome, treatment_values)

        $(estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates) = 
            $(statistical_estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)

        $(estimand)(outcome, treatment_values, treatment_confounders::Nothing, outcome_extra_covariates) = 
            $(causal_estimand)(outcome, treatment_values)

        $(estimand)(;outcome, treatment_values, treatment_confounders=nothing, outcome_extra_covariates=()) =
            $(estimand)(outcome, treatment_values, treatment_confounders, outcome_extra_covariates)

    end
    eval(ex)
end

CausalCMCompositeEstimands = Union{(eval(Symbol(:Causal, x)) for x in keys(ESTIMANDS_DOCS))...}

StatisticalCMCompositeEstimand = Union{(eval(Symbol(:Statistical, x)) for x in keys(ESTIMANDS_DOCS))...}

const AVAILABLE_ESTIMANDS = [x[1] for x ∈ ESTIMANDS_DOCS]

indicator_fns(Ψ::StatisticalCM) = Dict(values(Ψ.treatment_values) => 1.)

function indicator_fns(Ψ::StatisticalATE)
    case = []
    control = []
    for treatment in Ψ.treatment_values
        push!(case, treatment.case)
        push!(control, treatment.control)
    end
    return Dict(Tuple(case) => 1., Tuple(control) => -1.)
end

ncases(value, Ψ::StatisticalIATE) = sum(value[i] == Ψ.treatment_values[i].case for i in eachindex(value))

function indicator_fns(Ψ::StatisticalIATE)
    N = length(treatments(Ψ))
    key_vals = Pair[]
    for cf in Iterators.product((values(Ψ.treatment_values[T]) for T in treatments(Ψ))...)
        push!(key_vals, cf => float((-1)^(N - ncases(cf, Ψ))))
    end
    return Dict(key_vals...)
end

outcome_mean(Ψ::StatisticalCMCompositeEstimand) = ExpectedValue(Ψ.outcome, Tuple(union(Ψ.outcome_extra_covariates, keys(Ψ.treatment_confounders), (Ψ.treatment_confounders)...)))

outcome_mean_key(Ψ::StatisticalCMCompositeEstimand) = variables(outcome_mean(Ψ))

propensity_score(Ψ::StatisticalCMCompositeEstimand) = Tuple(ConditionalDistribution(T, Ψ.treatment_confounders[T]) for T in treatments(Ψ))

propensity_score_key(Ψ::StatisticalCMCompositeEstimand) = Tuple(variables(x) for x ∈ propensity_score(Ψ))

function get_relevant_factors(Ψ::StatisticalCMCompositeEstimand)
    outcome_model = outcome_mean(Ψ)
    treatment_factors = propensity_score(Ψ)
    return CMRelevantFactors(outcome_model, treatment_factors)
end

n_uniques_nuisance_functions(Ψ::StatisticalCMCompositeEstimand) = length(propensity_score(Ψ)) + 1

nuisance_functions_iterator(Ψ::StatisticalCMCompositeEstimand) =
    (propensity_score(Ψ)..., outcome_mean(Ψ))

function Base.show(io::IO, ::MIME"text/plain", Ψ::T) where T <: StatisticalCMCompositeEstimand 
    param_string = string(
        Base.typename(T).wrapper,
        "\n-----",
        "\nOutcome: ", Ψ.outcome,
        "\nTreatment: ", Ψ.treatment_values
    )
    println(io, param_string)
end

function treatment_specs_to_dict(treatment_values::NamedTuple{T, <:Tuple{Vararg{NamedTuple}}}) where T
    Dict(key => Dict(pairs(vals)) for (key, vals) in pairs(treatment_values))
end

treatment_specs_to_dict(treatment_values::NamedTuple) = Dict(pairs(treatment_values))

treatment_values(d::AbstractDict) = (;d...)
treatment_values(d) = d

confounders_values(key_value_iterable::Union{NamedTuple, Dict}, T) = unique_sorted_tuple(key_value_iterable[T])

confounders_values(iterable, T) = unique_sorted_tuple(iterable)

get_treatment_specs(treatment_specs::NamedTuple{names, }) where names = 
    NamedTuple{Tuple(sort(collect(names)))}(treatment_specs)

function get_treatment_specs(treatment_specs::NamedTuple{names, <:Tuple{Vararg{NamedTuple}}}) where names
    case_control = ((case=v[:case], control=v[:control]) for v in values(treatment_specs))
    treatment_specs = (;zip(keys(treatment_specs), case_control)...)
    sorted_names = Tuple(sort(collect(names)))
    return NamedTuple{sorted_names}(treatment_specs)
end
    
get_treatment_specs(treatment_specs::AbstractDict) = 
    get_treatment_specs((;(key => treatment_values(val) for (key, val) in treatment_specs)...))

constructorname(T; prefix="TMLE.Causal") = replace(string(T), prefix => "")

treatment_confounders_to_dict(treatment_confounders::NamedTuple) = 
    Dict(key => collect(vals) for (key, vals) in pairs(treatment_confounders))

"""
    to_dict(Ψ::T) where T <: CausalCMCompositeEstimands

Converts Ψ to a dictionary that can be serialized.
"""
function to_dict(Ψ::T) where T <: CausalCMCompositeEstimands
    return Dict(
        :type => constructorname(T; prefix="TMLE.Causal"),
        :outcome => Ψ.outcome,
        :treatment_values => treatment_specs_to_dict(Ψ.treatment_values)
        )
end

"""
    to_dict(Ψ::T) where T <: StatisticalCMCompositeEstimand

Converts Ψ to a dictionary that can be serialized.
"""
function to_dict(Ψ::T) where T <: StatisticalCMCompositeEstimand
    return Dict(
        :type => constructorname(T; prefix="TMLE.Statistical"),
        :outcome => Ψ.outcome,
        :treatment_values => treatment_specs_to_dict(Ψ.treatment_values),
        :treatment_confounders => treatment_confounders_to_dict(Ψ.treatment_confounders),
        :outcome_extra_covariates => collect(Ψ.outcome_extra_covariates)
        )
end

identify(method, Ψ::StatisticalCMCompositeEstimand, scm) = Ψ

function identify(method::BackdoorAdjustment, causal_estimand::T, scm::SCM) where T<:CausalCMCompositeEstimands
    # Treatment confounders
    treatment_names = keys(causal_estimand.treatment_values)
    treatment_codes = [code_for(scm.graph, treatment) for treatment ∈ treatment_names]
    confounders_codes = scm.graph.graph.badjlist[treatment_codes]
    treatment_confounders = NamedTuple{treatment_names}(
        [[scm.graph.vertex_labels[w] for w in confounders_codes[i]] 
        for i in eachindex(confounders_codes)]
    )

    return statistical_type_from_causal_type(T)(;
        outcome=causal_estimand.outcome,
        treatment_values = causal_estimand.treatment_values,
        treatment_confounders = treatment_confounders,
        outcome_extra_covariates = method.outcome_extra_covariates
    )
end

function get_treatment_values(dataset, colname)
    counts = groupcount(skipmissing(Tables.getcolumn(dataset, colname)))
    sorted_counts = sort(collect(pairs(counts)), by = x -> x.second, rev=true)
    return first.(sorted_counts)
end

"""
    unique_treatment_values(dataset, colnames)

We ensure that the values are sorted by frequency to maximize 
the number of estimands passing the positivity constraint.
"""
unique_treatment_values(dataset, colnames) =
    (;(colname => get_treatment_values(dataset, colname) for colname in colnames)...)

"""
Generated from transitive treatment switches to create independent estimands.
"""
get_treatment_settings(::Union{typeof(ATE), typeof(IATE)}, treatments_unique_values::NamedTuple{names}) where names =
    NamedTuple{names}([collect(zip(vals[1:end-1], vals[2:end])) for vals in values(treatments_unique_values)])

get_treatment_settings(::typeof(CM), treatments_unique_values) = treatments_unique_values

get_treatment_setting(combo::Tuple{Vararg{Tuple}}) = [NamedTuple{(:control, :case)}(treatment_control_case) for treatment_control_case ∈ combo]

get_treatment_setting(combo) = collect(combo)

"""
If there is no dataset and the treatments_levels are a NamedTuple, then they are assumed correct.
"""
make_or_check_treatment_levels(treatments_levels::NamedTuple, dataset::Nothing) = treatments_levels

"""
If no dataset is provided, then a NamedTuple precising treatment levels is expected
"""
make_or_check_treatment_levels(treatments, dataset::Nothing) = 
    throw(ArgumenError("No dataset from which to infer treatment levels was provided. Either provide a `dataset` or a NamedTuple `treatments` e.g. (T=[0, 1, 2],)"))

"""
If a list of treatments is provided as well as a dataset then the treatment_levels are infered from it.
"""
make_or_check_treatment_levels(treatments, dataset) = unique_treatment_values(dataset, treatments)

"""
If a NamedTuple of treatments_levels is provided as well as a dataset then the treatment_levels are checked from the dataset.
"""
function make_or_check_treatment_levels(treatments_levels::NamedTuple, dataset)
    for (treatment, treatment_levels) in zip(keys(treatments_levels), values(treatments_levels))
        dataset_treatment_levels = Set(skipmissing(Tables.getcolumn(dataset, treatment)))
        missing_levels = setdiff(treatment_levels, dataset_treatment_levels)
        length(missing_levels) == 0 || 
            throw(ArgumentError(string("Not all levels provided for treatment ", treatment, " were found in the dataset: ", missing_levels)))
    end
    return treatments_levels
end

function _factorialEstimand(
    constructor, 
    treatments_settings::NamedTuple{names}, outcome; 
    confounders=nothing,
    outcome_extra_covariates=nothing,
    freq_table=nothing,
    positivity_constraint=nothing,
    verbosity=1
    ) where names
    components = []
    for combo ∈ Iterators.product(values(treatments_settings)...)
        Ψ = constructor(
            outcome=outcome,
            treatment_values=NamedTuple{names}(get_treatment_setting(combo)),
            treatment_confounders = confounders,
            outcome_extra_covariates=outcome_extra_covariates
        )
        if satisfies_positivity(Ψ, freq_table; positivity_constraint=positivity_constraint)
            push!(components, Ψ)
        else
            verbosity > 0 && @warn("Sub estimand", Ψ, " did not pass the positivity constraint, skipped.")
        end
    end
    return ComposedEstimand(joint_estimand, Tuple(components))
end

"""
    factorialEstimand(
        constructor::Union{typeof(CM), typeof(ATE), typeof(IATE)},
        treatments, outcome; 
        confounders=nothing,
        dataset=nothing, 
        outcome_extra_covariates=(),
        positivity_constraint=nothing,
        freq_table=nothing,
        verbosity=1
    )

Generates a factorial `ComposedEstimand` with components of type `constructor` (CM, ATE, IATE). 

For the ATE and the IATE, the generated components are restricted to the Cartesian Product of single treatment levels transitions.
For example, consider two treatment variables T₁ and T₂ each taking three possible values (0, 1, 2). 
For each treatment variable, the single treatment levels transitions are defined by (0 → 1, 1 → 2). 
Then, the Cartesian Product of these transitions is taken, resulting in a 2 x 2 = 4 dimensional joint estimand:

- (T₁: 0 → 1, T₂: 0 → 1)
- (T₁: 0 → 1, T₂: 1 → 2)
- (T₁: 1 → 2, T₂: 0 → 1)
- (T₁: 1 → 2, T₂: 1 → 2)

# Return

A `ComposedEstimand` with causal or statistical components.

# Args

- `constructor`: CM, ATE or IATE.
- `treatments`: A NamedTuple of treatment levels (e.g. `(T=(0, 1, 2),)`) or a treatment iterator, then a dataset must be provided to infer the levels from it.
- `outcome`: The outcome variable.
- `confounders=nothing`: The generated components will inherit these confounding variables. If `nothing`, causal estimands are generated.
- `outcome_extra_covariates=()`: The generated components will inherit these `outcome_extra_covariates`.
- `dataset`: An optional dataset to enforce a positivity constraint and infer treatment levels.
- `positivity_constraint=nothing`: Only components that pass the positivity constraint are added to the `ComposedEstimand`. A `dataset` must then be provided.
- `freq_table`: This is only to be used by `factorialEstimands` to avoid unecessary computations.
- `verbosity=1`: Verbosity level.

# Examples:

- An Average Treatment Effect with causal components:

```@example
factorialEstimand(ATE, (T₁ = (0, 1), T₂=(0, 1, 2)), :Y₁)
```

- An Average Interaction Effect with statistical components:

```@example
factorial(IATE, (T₁ = (0, 1, 2), T₂=(0, 1, 2)), :Y₁, confounders=[:W₁, :W₂])
```

- With a dataset, the treatment levels can be infered and a positivity constraint enforced:
Interactions:

```@example
factorialEstimand(ATE, [:T₁, :T₂], :Y₁, 
    confounders=[:W₁, :W₂], 
    dataset=dataset, 
    positivity_constraint=0.1
)
```

"""
function factorialEstimand(
    constructor::Union{typeof(CM), typeof(ATE), typeof(IATE)},
    treatments, outcome; 
    confounders=nothing,
    dataset=nothing, 
    outcome_extra_covariates=(),
    positivity_constraint=nothing,
    freq_table=nothing,
    verbosity=1
    )
    treatments_levels = make_or_check_treatment_levels(treatments, dataset)
    freq_table = freq_table !== nothing ? freq_table : get_frequency_table(positivity_constraint, dataset, keys(treatments_levels))
    treatments_settings = get_treatment_settings(constructor, treatments_levels)
    return  _factorialEstimand(
        constructor, treatments_settings, outcome; 
        confounders=confounders,
        outcome_extra_covariates=outcome_extra_covariates,
        freq_table=freq_table,
        positivity_constraint=positivity_constraint,
        verbosity=verbosity
        )
end

"""
factorialEstimands(
    constructor::Union{typeof(ATE), typeof(IATE)},
    dataset, treatments, outcomes; 
    confounders=nothing, 
    outcome_extra_covariates=(),
    positivity_constraint=nothing,
    verbosity=1
    )

Generates a `ComposedEstimand` for each outcome in `outcomes`. See `factorialEstimand`.
"""
function factorialEstimands(
    constructor::Union{typeof(CM), typeof(ATE), typeof(IATE)},
    treatments, outcomes; 
    dataset=nothing,
    confounders=nothing, 
    outcome_extra_covariates=(),
    positivity_constraint=nothing,
    verbosity=1
    )
    treatments_levels = make_or_check_treatment_levels(treatments, dataset)
    freq_table = get_frequency_table(positivity_constraint, dataset, keys(treatments_levels))
    treatments_settings = get_treatment_settings(constructor, treatments_levels)
    estimands = []
    for outcome in outcomes
        Ψ = _factorialEstimand(
            constructor, treatments_settings, outcome; 
            confounders=confounders,
            outcome_extra_covariates=outcome_extra_covariates,
            freq_table=freq_table,
            positivity_constraint=positivity_constraint,
            verbosity=verbosity-1
            )
        if length(Ψ.args) > 0
            push!(estimands, Ψ)
        else
            verbosity > 0 && @warn(string(
                "ATE for outcome, ", outcome, 
                " has no component passing the positivity constraint, skipped."
            ))
        end
    end
    return estimands
end

joint_levels(Ψ::StatisticalIATE) = Iterators.product(values(Ψ.treatment_values)...)

joint_levels(Ψ::StatisticalATE) =
    (Tuple(Ψ.treatment_values[T][c] for T ∈ keys(Ψ.treatment_values)) for c in (:case, :control))

joint_levels(Ψ::StatisticalCM) = (values(Ψ.treatment_values),)