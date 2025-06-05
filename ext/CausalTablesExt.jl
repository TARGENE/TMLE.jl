module CausalTablesExt

using CausalTables
using TMLE
using CategoricalArrays
using DataFrames

"""
    (estimator::Estimator)(Ψ::Estimand, ct::CausalTable; cache=Dict(), verbosity=1)

Estimates a causal estimand Ψ using `estimator` from data wrapped as a CausalTable object 
from the CausalTables.jl package. The causal estimand is automatically identified as a 
statistical estimand using the structure encoded in the CausalTable. 

"""
function (estimator::Union{TMLE.Plugin, TMLE.Ose, TMLE.Tmle})(Ψ::TMLE.CausalCMCompositeEstimands, ct::CausalTables.CausalTable; kwargs...)
    # Convert treatment columns of the CausalTable to CategoricalArray
    cleaned_data = DataFrame([(k => k ∈ ct.treatment ? categorical(v) : v) for (k,v) in pairs(ct.data)])

    # Create a StructuralCausalModel from the CausalTable and identify
    scm = SCM([k => v for (k, v) in pairs(ct.causes)])
    statisticalΨ = identify(Ψ, scm)

    # Call the estimator with the identified Estimand and the cleaned Table of data
    return estimator(statisticalΨ, cleaned_data; kwargs...)
end

end

