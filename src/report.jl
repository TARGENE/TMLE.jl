struct QueryReport
    query::NamedTuple
    influence_curve::Vector{Float64}
    estimate::Float64
    initial_estimate::Float64
end

function summary(report::QueryReport; tail=:both)
    stderr = standarderror(report.influence_curve)

    return (pvalue=pvalue(report.estimate, stderr, tail=tail), 
            confint=confinterval(report.estimate, stderr), 
            estimate=report.estimate, 
            stderror=stderr, 
            initial_estimate=report.initial_estimate, 
            mean_inf_curve=mean(report.influence_curve))
end


"""
    summaries(m::Machine{TMLEstimator}; tail=:both)

Returns the reported results.
"""
function summaries(m::Machine{TMLEstimator}; tail=:both)
    machinereport = report(m)

    outputs = []
    for (i, _) in enumerate(m.model.queries)
        queryfield = Symbol("queryreport_$i")
        queryreport = getfield(machinereport, queryfield)
        push!(outputs, summary(queryreport; tail=tail))
    end

    return outputs
end
