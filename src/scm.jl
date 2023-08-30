#####################################################################
###                  Structural Equation                          ###
#####################################################################

SelfReferringEquationError(outcome) = 
    ArgumentError(string("Variable ", outcome, " appears on both sides of the equation."))

"""
# Structural Equation / SE

## Constructors

- SE(outcome, parents)
- SE(;outcome, parents)

## Examples

eq = SE(:Y, [:T, :W])
eq = SE(outcome=:Y, parents=[:T, :W])

"""
struct StructuralEquation
    outcome::Symbol
    parents::Set{Symbol}
    function StructuralEquation(outcome, parents)
        outcome = Symbol(outcome)
        parents = Set(Symbol(x) for x in parents)
        outcome ∉ parents || throw(SelfReferringEquationError(outcome))
        return new(outcome, parents)
    end
end

const SE = StructuralEquation

SE(;outcome, parents) = SE(outcome, parents)

outcome(se::SE) = se.outcome
parents(se::SE) = se.parents

string_repr(eq::SE; subscript="") = string(outcome(eq), " = f", subscript, "(", join(parents(eq), ", "), ")")

Base.show(io::IO, ::MIME"text/plain", eq::SE) = println(io, string_repr(eq))

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
    factors::Dict
end

const SCM = StructuralCausalModel

SCM(;equations=Dict{Symbol, SE}(), factors=Dict()) = SCM(equations, factors)
SCM(equations::Vararg{SE}; factors=Dict()) = SCM(Dict(outcome(eq) => eq for eq in equations), factors)

equations(scm::SCM) = scm.equations
factors(scm::SCM) = scm.factors

parents(scm::StructuralCausalModel, outcome::Symbol) = 
    haskey(equations(scm), outcome) ? parents(equations(scm)[outcome]) : Set{Symbol}()

setequation!(scm::SCM, eq::SE) = scm.equations[outcome(eq)] = eq

getequation(scm::StructuralCausalModel, key::Symbol) = scm.equations[key]

get_conditional_distribution(scm::SCM, outcome::Symbol) = get_conditional_distribution(scm, outcome, parents(scm, outcome))
get_conditional_distribution(scm::SCM, outcome::Symbol, parents) = scm.factors[(outcome, Set(parents))]

NoAvailableConditionalDistributionError(scm, outcome, conditioning_set) = ArgumentError(string(
    "Could not find a conditional distribution for either : \n - The required: ", 
    cond_dist_string(outcome, conditioning_set), 
    "\n - The natural (that could be used as a template): ", 
    cond_dist_string(outcome, parents(scm, outcome)), 
    "\nSet at least one of them using `set_conditional_distribution!` before proceeding to estimation."
    ))

UsingNaturalDistributionLog(scm, outcome, conditioning_set) = string(
    "Could not retrieve a conditional distribution for the required: ", 
    cond_dist_string(outcome, conditioning_set),
    ".\n Will use the model class found in the natural candidate: ",
    cond_dist_string(outcome, parents(scm, outcome))
)

function get_or_set_conditional_distribution_from_natural!(scm::SCM, outcome::Symbol, conditioning_set::Set{Symbol}; verbosity=1)
    try 
        return get_conditional_distribution(scm, outcome, conditioning_set)
    catch KeyError
        try
            template_factor = get_conditional_distribution(scm, outcome)
            set_conditional_distribution!(scm, outcome, conditioning_set, template_factor.model)
            factor = get_conditional_distribution(scm, outcome, conditioning_set)
            verbosity > 0 && @info(UsingNaturalDistributionLog(scm, outcome, conditioning_set))
            return factor
        catch KeyError
            throw(NoAvailableConditionalDistributionError(scm, outcome, conditioning_set))
        end
    end
end

set_conditional_distribution!(scm::SCM, outcome::Symbol, model) =
    set_conditional_distribution!(scm, outcome, parents(scm, outcome), model)

function set_conditional_distribution!(scm::SCM, outcome::Symbol, parents, model)
    factor = ConditionalDistribution(outcome, parents, model)
    setfactor!(scm, factor)
end

setfactor!(scm::SCM, factor::DistributionFactor) =
    scm.factors[key(factor)] = factor

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
            upstream = union((parents(scm, v) for v in upstream)...)
        end
        isempty(upstream) && return false
    end
end

#####################################################################
###                  StaticConfoundedModel                        ###
#####################################################################

combine_outcome_parents(treatment, confounders, covariates::Nothing) = vcat(treatment, confounders)
combine_outcome_parents(treatment, confounders, covariates) = vcat(treatment, confounders, covariates)

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
    outcome, treatment, confounders; 
    covariates = nothing, 
    outcome_model = TreatmentTransformer() |> LinearRegressor(),
    treatment_model = LinearBinaryClassifier(),
    )
    Yeq = SE(outcome, combine_outcome_parents(treatment, confounders, covariates))
    Teq = SE(treatment, vcat(confounders))
    outcome_factor = ConditionalDistribution(Yeq.outcome, Yeq.parents, outcome_model)
    treatment_factor = ConditionalDistribution(Teq.outcome, Teq.parents, treatment_model)
    factors = Dict(
        key(outcome_factor) => outcome_factor,
        key(treatment_factor) => treatment_factor
    )
    return StructuralCausalModel(Yeq, Teq; factors)
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
    outcomes::AbstractVector, 
    treatments::AbstractVector, 
    confounders; 
    covariates = nothing, 
    outcome_model = TreatmentTransformer() |> LinearRegressor(),
    treatment_model = LinearBinaryClassifier()
    )
    Yequations = (SE(outcome, combine_outcome_parents(treatments, confounders, covariates)) for outcome in outcomes)
    Tequations = (SE(treatment, vcat(confounders)) for treatment in treatments)
    factors = Dict(
        (outcome(Yeq) => Dict(parents(Yeq) => outcome_model) for Yeq in Yequations)...,
        (outcome(Teq) => Dict(parents(Teq) => treatment_model) for Teq in Tequations)...
    )
    factors = Dict()
    for eq in Yequations
        factor = ConditionalDistribution(eq.outcome, eq.parents, outcome_model)
        factors[key(factor)] = factor
    end
    for eq in Tequations
        factor = ConditionalDistribution(eq.outcome, eq.parents, treatment_model)
        factors[key(factor)] = factor
    end
    return SCM(Yequations..., Tequations...; factors=factors)
end