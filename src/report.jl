struct Report
    target_name::Symbol
    query::Query
    influence_curve::Vector{Float64}
    estimate::Float64
    initial_estimate::Float64
    Report(target_name, query, influence_curve, estimate, initial_estimate) =
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
function briefreport(r::Report; tail=:both, level=0.95)
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

"""
    briefreport(mach::Machine{TMLEstimator}; tail=:both, alpha=0.05)

For a given Machine{<:TMLEstimator}, provides a summary of useful statistics for each query.

# Arguments:
    - mach: The fitted machine
    - tail: controls weither the test is single or two sided: eg :left, :right or :both
    - alpha: level of the test
"""
briefreport(mach::Machine{TMLEstimator}; tail=:both, level=0.95) =
    Tuple(briefreport(r, tail=tail, level=level) for r in queryreports(mach))

queryreportname(target_idx::Int, query_idx::Int) = 
    Symbol(string("target_", target_idx, "_query_", query_idx))

queryreport(mach::Machine{<:TMLEstimator}, target_idx::Int, query_idx::Int) =
    getfield(mach.report, queryreportname(target_idx, query_idx))

queryreports(mach::Machine{<:TMLEstimator}) = 
    Tuple(r for r in report(mach) if r isa Report)


# Simple Test
"""
    ztest(r::Report)

If the original data is i.i.d, the influence curve is Normally distributed
and its variance can be estimated by the sample variance over all samples.
We can then perform a Z-Test for a given Report object. It will test weither the measured 
effect size is significantly different from 0 under those assumptions.
""" 
ztest(r::Report) = 
    OneSampleZTest(r.estimate .+ r.influence_curve)

"""
    ztest(mach::Machine{<:TMLEstimator}, target_idx::Int, query_idx::Int)

Performs a Z-Test for the given target/query pair.
See also: ztest(r::Report)
"""
ztest(mach::Machine{<:TMLEstimator}, target_idx::Int, query_idx::Int) = ztest(queryreport(mach, target_idx, query_idx))
"""
    ztest(mach::Machine{<:TMLEstimator})

Performs a Z-Test for all target/query pairs fitted by the machine.
See also: ztest(r::Report)
"""
ztest(mach::Machine{<:TMLEstimator}) =
    Tuple((target_name=r.target_name, query_name=r.query.name, test_result=ztest(r)) 
        for r in queryreports(mach))

# Paired Test
ztest(r₁::Report, r₂::Report) =
    OneSampleZTest(r₁.influence_curve .+ r₁.estimate, r₂.influence_curve .+ r₂.estimate)
ztest(mach::Machine{<:TMLEstimator}, target_idx::Int, query_idx_pair::Pair{Int, Int}) =
    ztest(queryreport(mach, target_idx, query_idx_pair[1]), queryreport(mach, target_idx, query_idx_pair[2]))

