struct TMLEReport
    target_name::Symbol
    query::Query
    influence_curve::Vector{Float64}
    estimate::Float64
    initial_estimate::Float64
    TMLEReport(target_name, query, influence_curve, estimate, initial_estimate) =
        new(Symbol(target_name), query, influence_curve, estimate, initial_estimate)
end


"""
    briefreport(r::Report; tail=:both, alpha=0.05)

For a given Report, provides a summary of useful statistics.

# Arguments:
    - r: A query report, for instance extracted via `queryreport`
    - tail: controls weither the test is single or two sided: eg :left, :right or :both
    - alpha: level of the test
"""
function summarize(r::TMLEReport; tail=:both, level=0.95)
    testresult = ztest(r)
    return (target_name=r.target_name,
            query=r.query,
            pvalue=pvalue(testresult, tail=tail), 
            confint=confint(testresult, level=level, tail=tail), 
            estimate=r.estimate, 
            initial_estimate=r.initial_estimate, 
            stderror=testresult.stderr,
            mean_inf_curve=mean(r.influence_curve))
end

summarize(tmle_reports::Dict; tail=:both, level=0.95) =
    Dict((key, summarize(report; tail=tail, level=level)) for (key,report) in tmle_reports)


# Simple Test
"""
    ztest(r::Report)

If the original data is i.i.d, the influence curve is Normally distributed
and its variance can be estimated by the sample variance over all samples.
We can then perform a Z-Test for a given Report object. It will test weither the measured 
effect size is significantly different from 0 under those assumptions.
""" 
ztest(r::TMLEReport) = 
    OneSampleZTest(r.estimate .+ r.influence_curve)

ztest(tmle_reports::Dict) =
    Dict((key, ztest(report)) for (key,report) in tmle_reports)

# Paired Test
function ztest(r₁::TMLEReport, r₂::TMLEReport)
    μ₁ = mean(r₁.influence_curve.^2)
    μ₂ = mean(r₂.influence_curve.^2)
    μ₁₂ = mean(r₁.influence_curve .* r₂.influence_curve)

    return OneSampleZTest(r₁.estimate-r₂.estimate, sqrt(μ₁ + μ₂ - 2μ₁₂), size(r₁.influence_curve, 1))
end

