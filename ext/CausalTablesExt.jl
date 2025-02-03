module CausalTablesExt

using CausalTables
using TMLE
using CategoricalArrays

function (estimator::TMLE.Estimator)(Ψ::TMLE.Estimand, ct::CausalTables.CausalTable; kwargs...)
    # Convert treatment columns of the CausalTable to CategoricalArray
    cleaned_data = NamedTuple([(k => k ∈ ct.treatment ? CategoricalArrays.categorical(v) : v) for (k,v) in pairs(ct.data)])

    # Create a StructuralCausalModel from the CausalTable and identify
    scm = TMLE.SCM([k => v for (k, v) in pairs(ct.causes)])
    statisticalΨ = TMLE.identify(Ψ, scm)

    # Call the estimator with the identified Estimand and the cleaned Table of data
    return estimator(statisticalΨ, cleaned_data; kwargs...)
end

end

