#####################################################################
###                  Structural Equation                          ###
#####################################################################

# const AVAILABLE_SPECIFICATIONS = Set([:density, :mean])

SelfReferringEquationError(outcome) = 
    ArgumentError(string("Variable ", outcome, " appears on both sides of the equation."))

# UnavailableSpecificationError(key) = 
#     ArgumentError(string("Unavailable specification: ", key, ". Available specifications are: ", join(AVAILABLE_SPECIFICATIONS, ", "), "."))

# SpecificationShouldBeModelError(key) = 
#     ArgumentError(string("Specification ", key, " should be a MLJBase.Model."))

struct StructuralEquation
    outcome::Symbol
    parents::Vector{Symbol}
    model::Model
    function StructuralEquation(outcome, parents, model)
        outcome ∉ parents || throw(SelfReferringEquationError(outcome))
        # for (specname, model) ∈ zip(keys(specifications), specifications)
        #     specname ∈ AVAILABLE_SPECIFICATIONS || throw(UnavailableSpecificationError(specname))
        #     model isa Model || throw(SpecificationShouldBeModelError(specname))
        # end
        if model isa Nothing
            return new(outcome, parents)
        else
            return new(outcome, parents, model)
        end
    end
end

StructuralEquation(outcome, parents; model=nothing) = StructuralEquation(outcome, parents, model)

const SE = StructuralEquation

assign_model!(eq::SE, model::Nothing) = nothing
assign_model!(eq::SE, model::Model) = eq.model = model

outcome(se::SE) = se.outcome
parents(se::SE) = se.parents

#####################################################################
###                Structural Causal Model                        ###
#####################################################################

AlreadyAssignedError(key) = ArgumentError(string("Variable ", key, " is already assigned in the SCM."))

struct StructuralCausalModel
    equations::Dict{Symbol, StructuralEquation}
end

const SCM = StructuralCausalModel

StructuralCausalModel(equations::Vararg{SE}) = 
    StructuralCausalModel(Dict(outcome(eq) => eq for eq in equations))

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

parents(scm::StructuralCausalModel, key::Symbol) = parents(getproperty(scm, key))

#####################################################################
###                  StaticConfoundedModel                        ###
#####################################################################

vcat_covariates(covariates::Nothing, treatment, confounders) = vcat(treatment, confounders)
vcat_covariates(covariates, treatment, confounders) = vcat(covariates, treatment, confounders)

function StaticConfoundedModel(
    outcome::Symbol, treatment::Symbol, confounders::Union{Symbol, AbstractVector{Symbol}}; 
    covariates::Union{Nothing, Symbol, AbstractVector{Symbol}} = nothing, 
    outcome_spec = LinearRegressor(),
    treatment_spec = LinearBinaryClassifier()
    )
    Yeq = StructuralEquation(
        outcome, 
        vcat_covariates(covariates, treatment, confounders), 
        outcome_spec
    )
    Teq = StructuralEquation(
        treatment, 
        vcat(confounders), 
        treatment_spec
    )
    return StructuralCausalModel(Yeq, Teq)
end