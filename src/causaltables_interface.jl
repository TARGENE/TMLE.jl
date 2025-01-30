
function (estimator::Estimator)(Ψ::Estimand, ct::CausalTable; kwargs...)
    # Convert treatment columns of the CausalTable to CategoricalArray
    cleaned_data = NamedTuple([(k => k ∈ ct.treatment ? categorical(v) : v) for (k,v) in pairs(ct.data)])

    # Create a StructuralCausalModel from the CausalTable and identify
    scm = SCM([k => v for (k, v) in pairs(ct.causes)])
    statisticalΨ = identify(Ψ, scm)

    # Call the estimator with the identified Estimand and the cleaned Table of data
    return estimator(statisticalΨ, cleaned_data; kwargs...)
end

