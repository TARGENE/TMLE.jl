#####################################################################
###                  Structural Equation                          ###
#####################################################################
"""
# Structural Equation / SE

## Constructors

- SE(outcome, parents; model=nothing)
- SE(;outcome, parents, model=nothing)

## Examples

eq = SE(:Y, [:T, :W])
eq = SE(:Y, [:T, :W], model = LinearRegressor())

"""
mutable struct StructuralEquation
    outcome::Symbol
    parents::Vector{Symbol}
    model::Union{Model, Nothing}
    mach::Union{Machine, Nothing}
    function StructuralEquation(outcome, parents, model)
        outcome ∉ parents || throw(SelfReferringEquationError(outcome))
        return new(outcome, parents, model, nothing)
    end
end

const SE = StructuralEquation

SE(outcome, parents; model=nothing) = SE(outcome, parents, model)
SE(;outcome, parents, model=nothing) = SE(outcome, parents, model)

SelfReferringEquationError(outcome) = 
    ArgumentError(string("Variable ", outcome, " appears on both sides of the equation."))

NoModelError(eq::SE) = ArgumentError(string("It seems the following structural equation needs to be fitted.\n",
    " Please provide a suitable model for it :\n\t⋆ ", eq))

fit_message(eq::SE) = string("Fitting Structural Equation corresponding to variable ", outcome(eq), ".")


equation_inputs(eq::SE, input_variables::Nothing) = parents(eq)
equation_inputs(eq::SE, input_variables) = input_variables

"""
    MLJBase.fit!(eq::SE, dataset; input_variables=nothing, verbosity=1, cache=true, force=false)

Fits the outcome's Structural Equation using the dataset with inputs variables given by either:

- The variables corresponding to parents(eq) if `input_variables`= nothing
- The alternative variables provided by `input_variables`.

Extra keyword arguments are:

- cache: Controls whether the associated MLJ.Machine will cache data.
- force: Controls whether to force the associated MLJ.Machine to refit even if neither the model or data has changed.
- verbosity: Controls the verbosity level
"""
function MLJBase.fit!(eq::SE, dataset; input_variables=nothing, verbosity=1, cache=true, force=false)
    eq.model !== nothing || throw(NoModelError(eq))
    # Fit when: never fitted OR model has changed OR inputs have changed
    dofit = eq.mach === nothing || eq.model != eq.mach.model
    if !dofit
        if isdefined(eq.mach, :data)
            dofit = Tables.columnnames(eq.mach.data[1]) !== Tuple(equation_inputs(eq, input_variables))
        else
            dofit = true
        end
    end

    if dofit
        verbosity >= 1 && @info(fit_message(eq))
        input_colnames = equation_inputs(eq, input_variables)
        data = nomissing(dataset, vcat(input_colnames, outcome(eq)))
        X = selectcols(data, input_colnames)
        y = Tables.getcolumn(data, outcome(eq))
        mach = machine(eq.model, X, y, cache=cache)
        MLJBase.fit!(mach, verbosity=verbosity-1)
        eq.mach = mach
    # Also refit if force is true
    else 
        verbosity >= 1 && force === true && @info(fit_message(eq))
        MLJBase.fit!(eq.mach, verbosity=verbosity-1, force=force)
    end
end

reset!(eq::SE) = eq.mach = nothing

function string_repr(eq::SE; subscript="")
    eq_string = string(eq.outcome, " = f", subscript, "(", join(eq.parents, ", "), ")")
    if eq.model !== nothing
        eq_string = string(eq_string, ", ", nameof(typeof(eq.model)), ", fitted=", eq.mach !== nothing)
    end
    return eq_string
end

Base.show(io::IO, ::MIME"text/plain", eq::SE) = println(io, string_repr(eq))

assign_model!(eq::SE, model::Nothing) = nothing
assign_model!(eq::SE, model::Model) = eq.model = model

outcome(se::SE) = se.outcome
parents(se::SE) = se.parents

#####################################################################
###                Structural Causal Model                        ###
#####################################################################

AlreadyAssignedError(key) = ArgumentError(string("Variable ", key, " is already assigned in the SCM."))

"""
# Structural Causal Model / SCM

## Constructors

SCM(;equations=Dict{Symbol, SE}())
SCM(equations::Vararg{SE})

## Examples

scm = SCM(
    SE(:Y, [:T, :W, :C]),
    SE(:T, [:W])
)

"""
struct StructuralCausalModel
    equations::Dict{Symbol, StructuralEquation}
end

const SCM = StructuralCausalModel

SCM(;equations=Dict{Symbol, SE}()) = SCM(equations)
SCM(equations::Vararg{SE}) = 
    SCM(Dict(outcome(eq) => eq for eq in equations))


equations(scm::SCM) = scm.equations

function string_repr(scm::SCM)
    scm_string = """
    Structural Causal Model:
    -----------------------
    """
    for (index, (_, eq)) in enumerate(scm.equations)
        digits = split(string(index), "")
        subscript = join(Meta.parse("'\\U0208$digit'") for digit in digits)
        scm_string = string(scm_string, string_repr(eq;subscript=subscript), "\n")
    end
    return scm_string
end

Base.show(io::IO, ::MIME"text/plain", scm::SCM) = println(io, string_repr(scm))


function Base.push!(scm::SCM, eq::SE)
    key = outcome(eq)
    scm[key] = eq
end

function Base.setindex!(scm::SCM, eq::SE, key::Symbol)
    if haskey(scm.equations, key)
        throw(AlreadyAssignedError(key))
    end
    scm.equations[key] = eq
end

Base.getindex(scm::SCM, key::Symbol) = scm.equations[key]

function Base.getproperty(scm::StructuralCausalModel, key::Symbol)
    hasfield(StructuralCausalModel, key) && return getfield(scm, key)
    return scm.equations[key]
end

parents(scm::StructuralCausalModel, key::Symbol) = 
    haskey(equations(scm), key) ? parents(scm[key]) : Symbol[]

function MLJBase.fit!(scm::SCM, dataset; verbosity=1, cache=true, force=false)
    for eq in values(equations(scm))
        fit!(eq, dataset; verbosity=verbosity, cache=cache, force=force)
    end
end

function reset!(scm::SCM)
    for eq in values(equations(scm))
        reset!(eq)
    end
end

"""
    is_upstream(var₁, var₂, scm::SCM)

Checks whether var₁ is upstream of var₂ in the SCM.
"""
function is_upstream(var₁, var₂, scm::SCM)
    upstream = parents(scm, var₂)
    while true
        if var₁ ∈ upstream
            return true
        else
            upstream = vcat((parents(scm, v) for v in upstream)...)
        end
        isempty(upstream) && return false
    end
end

function upstream_variables(var, scm::SCM)
    upstream = parents(scm, var)
    current_upstream = upstream
    while true
        current_upstream = vcat((parents(scm, v) for v in current_upstream)...)
        union!(upstream, current_upstream)
        isempty(current_upstream) && return upstream
    end
end
#####################################################################
###                  StaticConfoundedModel                        ###
#####################################################################

vcat_covariates(treatment, confounders, covariates::Nothing) = vcat(treatment, confounders)
vcat_covariates(treatment, confounders, covariates) = vcat(treatment, confounders, covariates)

"""
    StaticConfoundedModel(
        outcome::Symbol, treatment::Symbol, confounders::Union{Symbol, AbstractVector{Symbol}}; 
        covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
        outcome_model = TreatmentTransformer() |> LinearRegressor(),
        treatment_model = LinearBinaryClassifier()
    )

Defines a classic Structural Causal Model with one outcome, one treatment, 
a set of confounding variables and optional covariates influencing the outcome only.

The `outcome_model` and `treatment_model` define the relationship between 
the outcome (resp. treatment) and their ancestors.
"""
function StaticConfoundedModel(
    outcome::Symbol, treatment::Symbol, confounders::Union{Symbol, AbstractVector{Symbol}}; 
    covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
    outcome_model = TreatmentTransformer() |> LinearRegressor(),
    treatment_model = LinearBinaryClassifier()
    )
    Yeq = SE(
        outcome, 
        vcat_covariates(treatment, confounders, covariates), 
        outcome_model
    )
    Teq = SE(
        treatment, 
        vcat(confounders), 
        treatment_model
    )
    return StructuralCausalModel(Yeq, Teq)
end

"""
    StaticConfoundedModel(
        outcomes::Vector{Symbol}, 
        treatments::Vector{Symbol}, 
        confounders::Union{Symbol, AbstractVector{Symbol}}; 
        covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
        outcome_model = TreatmentTransformer() |> LinearRegressor(),
        treatment_model = LinearBinaryClassifier()
    )

Defines a classic Structural Causal Model with multiple outcomes, multiple treatments, 
a set of confounding variables and optional covariates influencing the outcomes only.

All treatments are assumed to be direct parents of all outcomes. The confounding variables 
are shared for all treatments.

The `outcome_model` and `treatment_model` define the relationships between 
the outcomes (resp. treatments) and their ancestors.
"""
function StaticConfoundedModel(
    outcomes::Vector{Symbol}, 
    treatments::Vector{Symbol}, 
    confounders::Union{Symbol, AbstractVector{Symbol}}; 
    covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
    outcome_model = TreatmentTransformer() |> LinearRegressor(),
    treatment_model = LinearBinaryClassifier()
    )
    Yequations = (SE(outcome, vcat_covariates(treatments, confounders, covariates), outcome_model) for outcome in outcomes)
    Tequations = (SE(treatment, vcat(confounders), treatment_model) for treatment in treatments)
    return SCM(Yequations..., Tequations...)
end