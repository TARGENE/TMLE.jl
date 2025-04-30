#####################################################################
#####                 CausalStratifiedCV                         #####
#####################################################################

mutable struct CausalStratifiedCV <: MLJBase.ResamplingStrategy
    resampling::StratifiedCV
    treatment_variables::Vector{Symbol}
    CausalStratifiedCV(resampling) = new(resampling, Symbol[])
end

"""
    CausalStratifiedCV(;resampling=StratifiedCV())

Applies a stratified cross-validation strategy based on both treatments and outcome (if it is Finite) variables.
"""
CausalStratifiedCV(;resampling=StratifiedCV()) = CausalStratifiedCV(resampling)

function MLJBase.fit!(resampling::CausalStratifiedCV, Ψ, dataset)
    empty!(resampling.treatment_variables)
    append!(resampling.treatment_variables, treatments(Ψ))
end

update_stratification_col!(stratification_col::AbstractVector, col::AbstractVector) =
    stratification_col .*= string.(col, "_")

function update_stratification_col_if_finite!(stratification_col, col)
    if autotype(col) <: Union{Missing, Finite}
        update_stratification_col!(stratification_col, col)
    end
end

function aggregate_features!(stratification_col, columnnames, X)
    for colname in columnnames
        update_stratification_col_if_finite!(
            stratification_col, 
            Tables.getcolumn(X, colname)
        )
    end
end

"""
    MLJBase.train_test_pairs(resampling::CausalStratifiedCV, rows, X, y)

Constructs a new column used for stratification. This column is a combination of the treatment variables and potentilly 
the outcome variable if it is finite.
"""
function MLJBase.train_test_pairs(resampling::CausalStratifiedCV, rows, X, y)
    stratification_col = fill("", nrows(X))
    aggregate_features!(stratification_col, resampling.treatment_variables, X)
    update_stratification_col_if_finite!(stratification_col, y)
    return MLJBase.train_test_pairs(resampling.resampling, rows, X, categorical(stratification_col))
end

#####################################################################
#####                     TMLE  Interface                       #####
#####################################################################

"""
Default fit does nothing.
"""
MLJBase.fit!(resampling::ResamplingStrategy, Ψ, dataset) = nothing

"""
    get_train_validation_indices(resampling::ResamplingStrategy, Ψ, dataset)

This function gets called within the estimation process. It introduces a `fit!` call to the resampling strategy, 
to adapt the training and validation pairs based on the treatment variables defined in the estimand.
"""
function get_train_validation_indices(resampling::ResamplingStrategy, Ψ, dataset)
    MLJBase.fit!(resampling, Ψ, dataset)
    return MLJBase.train_test_pairs(
        resampling,
        1:nrows(dataset),
        dataset, 
        dataset[!, Ψ.outcome]
    )
end

get_train_validation_indices(resampling::Nothing, Ψ, dataset) = nothing