#####################################################################
###                    Abstract Estimand                          ###
#####################################################################
"""
A Estimand is a functional on distribution space Ψ: ℳ → ℜ. 
"""
abstract type Estimand end

treatments(Ψ::Estimand) = collect(keys(Ψ.treatment))
treatments(dataset, Ψ::Estimand) = selectcols(dataset, treatments(Ψ))

confounders(dataset, Ψ::Estimand) = (;(T => selectcols(dataset, keys(Ψ.scm[T].mach.data[1])) for T in treatments(Ψ))...)

AbsentLevelError(treatment_name, key, val, levels) = ArgumentError(string(
    "The treatment variable ", treatment_name, "'s, '", key, "' level: '", val,
    "' in Ψ does not match any level in the dataset: ", levels))

AbsentLevelError(treatment_name, val, levels) = ArgumentError(string(
    "The treatment variable ", treatment_name, "'s, level: '", val,
    "' in Ψ does not match any level in the dataset: ", levels))

"""
    check_treatment_settings(settings::NamedTuple, levels, treatment_name)

Checks the case/control values defining the treatment contrast are present in the dataset levels. 

Note: This method is for estimands like the ATE or IATE that have case/control treatment settings represented as 
`NamedTuple`.
"""
function check_treatment_settings(settings::NamedTuple, levels, treatment_name)
    for (key, val) in zip(keys(settings), settings) 
        any(val .== levels) || 
            throw(AbsentLevelError(treatment_name, key, val, levels))
    end
end

"""
    check_treatment_settings(setting, levels, treatment_name)

Checks the value defining the treatment setting is present in the dataset levels. 

Note: This is for estimands like the CM that do not have case/control treatment settings 
and are represented as simple values.
"""
function check_treatment_settings(setting, levels, treatment_name)
    any(setting .== levels) || 
            throw(
                AbsentLevelError(treatment_name, setting, levels))
end

"""
    check_treatment_levels(Ψ::Estimand, dataset)

Makes sure the defined treatment levels are present in the dataset.
"""
function check_treatment_levels(Ψ::Estimand, dataset)
    for treatment_name in treatments(Ψ)
        treatment_levels = levels(Tables.getcolumn(dataset, treatment_name))
        treatment_settings = getproperty(Ψ.treatment, treatment_name)
        check_treatment_settings(treatment_settings, treatment_levels, treatment_name)
    end
end

"""
Function used to sort estimands for optimal estimation ordering.
"""
function estimand_key end

"""
Retrieves the relevant factors of the distribution to be fitted.
"""
relevant_factors(Ψ::Estimand; adjustment_method::AdjustmentMethod)

"""
    optimize_ordering!(estimands::Vector{<:Estimand})

Optimizes the order of the `estimands` to maximize reuse of 
fitted equations in the associated SCM.
"""
optimize_ordering!(estimands::Vector{<:Estimand}) = sort!(estimands, by=estimand_key)

"""
    optimize_ordering(estimands::Vector{<:Estimand})

See [`optimize_ordering!`](@ref)
"""
optimize_ordering(estimands::Vector{<:Estimand}) = sort(estimands, by=estimand_key)