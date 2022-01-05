struct QueryReport
    query::NamedTuple
    influence_curve::Vector{Float64}
    estimate::Float64
    initial_estimate::Float64
end

influencecurve(covariate, y, observed_fluct, ct_fluct, estimate) = 
    covariate .* (float(y) .- observed_fluct) .+ ct_fluct .- estimate
    
standarderror(x) = sqrt(var(x)/nrows(x))

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


"""
    pvalue(m::Machine{TMLEstimator})

Computes the p-value associated with the estimated quantity.
"""
function pvalue(estimate, stderror; tail=:both)
    x = estimate/stderror

    dist = Normal(0, 1)
    if tail == :both
        min(2 * min(cdf(dist, x), ccdf(dist, x)), 1.0)
    elseif tail == :left
        cdf(dist, x)
    elseif tail == :right
        ccdf(dist, x)
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

"""
    confinterval(m::Machine{TMLEstimator})

Provides a 95% confidence interval for the true quantity of interest.
"""
function confinterval(estimate, stderror)
    return (estimate - 1.96stderror, estimate + 1.96stderror)
end