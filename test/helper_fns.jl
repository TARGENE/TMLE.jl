using MLJ


function asymptotics(estimator, problem_fn, rng, Ns)
    abs_mean_rel_errors = []
    abs_vars = []
    ATE = 0
    for n in Ns
        estimates_at_n = []
        for i in 1:10
            T, W, y, ATE = problem_fn(rng; n=n)
            mach = machine(estimator, T, W, y)
            fit!(mach, verbosity=0)
            fp = fitted_params(mach).R.fitresult
            push!(estimates_at_n, fp.estimate)
        end
        push!(abs_mean_rel_errors, 100mean([abs((x-ATE)/ATE) for x in estimates_at_n]))
        push!(abs_vars, var(estimates_at_n))
    end
    return abs_mean_rel_errors, abs_vars
end