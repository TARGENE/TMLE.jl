struct Report
    target_name::Symbol
    query::Query
    influence_curve::Vector{Float64}
    estimate::Float64
    initial_estimate::Float64
    Report(target_name, query, influence_curve, estimate, initial_estimate) =
        new(Symbol(target_name), query, influence_curve, estimate, initial_estimate)
end


influencecurve(covariate, y, observed_fluct, ct_fluct, estimate) = 
    covariate .* (float(y) .- observed_fluct) .+ ct_fluct .- estimate


"""
    briefreport(qr::Report; tail=:both, alpha=0.05)

For a given Report, provides a summary of useful statistics.

# Arguments:
    - qr: A query report, for instance extracted via `getqueryreport`
    - tail: controls weither the test is single or two sided: eg :left, :right or :both
    - alpha: level of the test
"""
function briefreport(qr::Report; tail=:both, level=0.95)
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
    Tuple(briefreport(qr, tail=tail, level=level) for qr in getqueryreports(mach))

queryreportname(target_idx::Int, query_idx::Int) = 
    Symbol(string("target_", target_idx, "_query_", query_idx))

getqueryreport(mach::Machine{<:TMLEstimator}, target_idx::Int, query_idx::Int) =
    getfield(mach.report, queryreportname(target_idx, query_idx))

getqueryreports(mach::Machine{<:TMLEstimator}) = 
    Tuple(qr for qr in report(mach) if qr isa Report)


# Simple Test   
ztest(qr::Report) = 
    OneSampleZTest(qr.estimate .+ qr.influence_curve)
ztest(mach::Machine{<:TMLEstimator}, target_idx::Int, query_idx::Int) = ztest(getqueryreport(mach, target_idx, query_idx))
ztest(mach::Machine{<:TMLEstimator}) =
    Tuple(ztest(qr) for qr in getqueryreports(mach))

# Paired Test
ztest(qr₁::Report, qr₂::Report) =
    OneSampleZTest(qr₁.influence_curve .+ qr₁.estimate, qr₂.influence_curve .+ qr₂.estimate)
ztest(mach::Machine{<:TMLEstimator}, target_idx::Int, query_idx_pair::Pair{Int, Int}) =
    ztest(getqueryreport(mach, target_idx, query_idx_pair[1]), getqueryreport(mach, target_idx, query_idx_pair[2]))

