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
If no complete_rows are provided, the validation indices are not remapped and are returned as is (for CTMLE mode).
"""
remap_validation_indices(src_complete_rows_ids::Nothing, src_train_validation_indices) = src_train_validation_indices

"""
This function maps the validation indices from the initial dataset to the fluctuation dataset.
It is based on the fact that `src_complete_rows_ids` was obtained from `findall(completecases(initial_factors_dataset))` 
which is also used to build the fluctuation dataset.
"""
function remap_validation_indices(src_complete_rows_ids, src_train_validation_indices)
    src_to_dest = Dict(src_id => dest_id for (dest_id, src_id) in enumerate(src_complete_rows_ids))
    remap(idx) = [src_to_dest[src_id] for src_id in idx if haskey(src_to_dest, src_id)]
    return [(src_train_idx, remap(src_val_idx)) for (src_train_idx, src_val_idx) in src_train_validation_indices]
end

"""
Default fit does nothing.
"""
MLJBase.fit!(resampling::ResamplingStrategy, Ψ, dataset) = nothing

default_resampling(collaborative_strategy::Nothing) = nothing

default_resampling(collaborative_strategy) = CausalStratifiedCV()

"""
    get_train_validation_indices(resampling::ResamplingStrategy, Ψ, dataset, complete_rows)

The particularity of these indices is that:
- The training indices are to be used on the initial_factors_dataset
- The validation indices are to be used on the fluctuation_dataset

As such, the validation indices are remapped to the fluctuation dataset
"""
function get_train_validation_indices(resampling::ResamplingStrategy, Ψ, dataset; complete_rows=nothing)
    MLJBase.fit!(resampling, Ψ, dataset)
    train_validation_indices = MLJBase.train_test_pairs(
        resampling,
        1:nrows(dataset),
        dataset, 
        dataset[!, Ψ.outcome]
    )
    return remap_validation_indices(complete_rows, train_validation_indices)
end
    
"""
    get_train_validation_indices(resampling::ResamplingStrategy, collaborative_strategy, Ψ, initial_factors_dataset, fluctuation_dataset, complete_rows)

In the case of a collaborative strategy, the train and validation pairs are built from the complete case `fluctuation_dataset`.
"""
get_train_validation_indices(
    resampling::ResamplingStrategy, 
    collaborative_strategy, 
    Ψ, 
    initial_factors_dataset, 
    fluctuation_dataset, 
    complete_rows
    ) = get_train_validation_indices(resampling, Ψ, fluctuation_dataset; complete_rows=nothing)

"""
    get_train_validation_indices(resampling::ResamplingStrategy, collaborative_strategy::Nothing, Ψ, initial_factors_dataset, fluctuation_dataset, complete_rows)

When there is no collaborative strategy, the train and validation pairs are built from the `initial_factors_dataset` and the `complete_rows`.
"""
get_train_validation_indices(
    resampling::ResamplingStrategy, 
    collaborative_strategy::Nothing, 
    Ψ, 
    initial_factors_dataset, 
    fluctuation_dataset, 
    complete_rows
    ) = get_train_validation_indices(resampling, Ψ, initial_factors_dataset, complete_rows=complete_rows)

"""
When there is no resampling, regardless of the collaborative strategy, the train and validation pairs are just nothing.
"""
get_train_validation_indices(
    resampling::Nothing, 
    collaborative_strategy, 
    Ψ, 
    initial_factors_dataset, 
    fluctuation_dataset, 
    complete_rows
    ) = nothing
