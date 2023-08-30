
abstract type AdjustmentMethod end

struct BackdoorAdjustment <: AdjustmentMethod
    outcome_extra::Vector{Symbol}
end

"""
    BackdoorAdjustment(;outcome_extra=[])

The adjustment set for each treatment variable is simply the set of direct parents in the 
associated structural model.

`outcome_extra` are optional additional variables that can be used to fit the outcome model 
in order to improve inference.
"""
BackdoorAdjustment(;outcome_extra=[]) = BackdoorAdjustment(outcome_extra)

confounders(::BackdoorAdjustment, Ψ::Estimand) = 
    union((parents(Ψ.scm, treatment) for treatment in treatments(Ψ))...)

outcome_parents(adjustment_method::BackdoorAdjustment, Ψ::Estimand) =
    Set(vcat(
        treatments(Ψ), 
        confounders(adjustment_method, Ψ)..., 
        adjustment_method.outcome_extra
    ))

