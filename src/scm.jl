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
    StructuralCausalModel(equations) = new(equations, Dict())
end

const SCM = StructuralCausalModel

SCM(;equations=Dict{Symbol, SE}()) = SCM(equations)
SCM(equations::Vararg{SE}) = SCM(Dict(outcome(eq) => eq for eq in equations))

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

function get_or_set_conditional_distribution_from_natural!(scm::SCM, outcome::Symbol, conditioning_set::Set{Symbol}; resampling=nothing, verbosity=1)
    try 
        # A factor has already been setup
        factor =  get_conditional_distribution(scm, outcome, conditioning_set)
        # If the resampling strategies also match then we can reuse the already set factor
        new_factor = ConditionalDistribution(outcome, conditioning_set, factor.model, resampling=resampling)
        if match(factor, new_factor)
            return factor
        else
            setfactor!(scm, new_factor)
            return new_factor
        end
    catch KeyError
        # A factor has not already been setup
        try
            # A natural factor can be used as a template
            template_factor = get_conditional_distribution(scm, outcome)
            new_factor = set_conditional_distribution!(scm, outcome, conditioning_set, template_factor.model, resampling)
            verbosity > 0 && @info(UsingNaturalDistributionLog(scm, outcome, conditioning_set))
            return new_factor
        catch KeyError
            throw(NoAvailableConditionalDistributionError(scm, outcome, conditioning_set))
        end
    end
end

set_conditional_distribution!(scm::SCM, outcome::Symbol, model, resampling) =
    set_conditional_distribution!(scm, outcome, parents(scm, outcome), model, resampling)

function set_conditional_distribution!(scm::SCM, outcome::Symbol, parents, model, resampling)
    factor = ConditionalDistribution(outcome, parents, model, resampling=resampling)
    return setfactor!(scm, factor)
end

function setfactor!(scm::SCM, factor)
    scm.factors[key(factor)] = factor
    return factor
end

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

VariableNotAChildInSCMError(variable) = ArgumentError(string("Variable ", variable, " is not associated with a Structural Equation in the SCM."))
TreatmentMustBeInOutcomeParentsError(variable) = ArgumentError(string("Treatment variable ", variable, " must be a parent of the outcome."))

function check_parameter_against_scm(scm::SCM, outcome, treatment)
    eqs = equations(scm)
    haskey(eqs, outcome) || throw(VariableNotAChildInSCMError(outcome))
    for treatment_variable in keys(treatment)
        haskey(eqs, treatment_variable) || throw(VariableNotAChildInSCMError(treatment_variable))
        is_upstream(treatment_variable, outcome, scm) || throw(TreatmentMustBeInOutcomeParentsError(treatment_variable))
    end
end

#####################################################################
###                  StaticConfoundedModel                        ###
#####################################################################

combine_outcome_parents(treatment, confounders, ::Nothing) = vcat(treatment, confounders)
combine_outcome_parents(treatment, confounders, covariates) = vcat(treatment, confounders, covariates)

"""
    StaticConfoundedModel(
        outcome::Symbol, treatment::Symbol, confounders::Union{Symbol, AbstractVector{Symbol}}; 
        covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing
    )

Defines a classic Structural Causal Model with one outcome, one treatment, 
a set of confounding variables and optional covariates influencing the outcome only.
"""
function StaticConfoundedModel(
    outcome, treatment, confounders; 
    covariates = nothing
    )
    Yeq = SE(outcome, combine_outcome_parents(treatment, confounders, covariates))
    Teq = SE(treatment, vcat(confounders))
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
"""
function StaticConfoundedModel(
    outcomes::AbstractVector, 
    treatments::AbstractVector, 
    confounders; 
    covariates = nothing
    )
    Yequations = (SE(outcome, combine_outcome_parents(treatments, confounders, covariates)) for outcome in outcomes)
    Tequations = (SE(treatment, vcat(confounders)) for treatment in treatments)
    return SCM(Yequations..., Tequations...)
end