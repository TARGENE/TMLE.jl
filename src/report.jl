struct QueryReport
    query::NamedTuple
    influence_curve::Vector{Float64}
    estimate::Float64
    initial_estimate::Float64
end

influencecurve(covariate, y, observed_fluct, ct_fluct, estimate) = 
    covariate .* (float(y) .- observed_fluct) .+ ct_fluct .- estimate


"""
    briefreport(qr::QueryReport; tail=:both, alpha=0.05)

For a given QueryReport, provides a summary of useful statistics.

# Arguments:
    - qr: A query report, for instance extracted via `getqueryreport`
    - tail: controls weither the test is single or two sided: eg :left, :right or :both
    - alpha: level of the test
"""
function briefreport(qr::QueryReport; tail=:both, level=0.95)
    testresult = ztest(qr)
    return (query=qr.query,
            pvalue=pvalue(testresult, tail=tail), 
            confint=confint(testresult, level=level, tail=tail), 
            estimate=qr.estimate, 
            initial_estimate=qr.initial_estimate, 
            stderror=testresult.stderr,
            mean_inf_curve=mean(qr.influence_curve))
end

"""
    briefreport(mach::Machine{TMLEstimator}; tail=:both, alpha=0.05)

For a given Machine{<:TMLEstimator}, provides a summary of useful statistics for each query.

# Arguments:
    - mach: The fitted machine
    - tail: controls weither the test is single or two sided: eg :left, :right or :both
    - alpha: level of the test
"""
briefreport(mach::Machine{TMLEstimator}; tail=:both, level=0.95) =
    Tuple(briefreport(getqueryreport(mach, i), tail=tail, level=level) 
            for (i, q) in enumerate(mach.model.queries))


queryreportname(i::Int) = Symbol("queryreport_$i")

getqueryreport(mach::Machine{<:TMLEstimator}, i::Int) =
    getfield(mach.report, queryreportname(i))

# Simple Test   
ztest(qr::QueryReport) = 
    OneSampleZTest(qr.estimate .+ qr.influence_curve)
ztest(mach::Machine{<:TMLEstimator}, arg::Int) = ztest(getqueryreport(mach, arg))
ztest(mach::Machine{<:TMLEstimator}, args::Vararg{Int}) =
    Tuple(ztest(getqueryreport(mach, arg)) for arg in args)

# Paired Test
ztest(qr₁::QueryReport, qr₂::QueryReport) =
    OneSampleZTest(qr₁.influence_curve .+ qr₁.estimate, qr₂.influence_curve .+ qr₂.estimate)
ztest(mach::Machine{<:TMLEstimator}, pair::Pair{Int, Int}) =
    ztest(getqueryreport(mach, pair[1]), getqueryreport(mach, pair[2]))

