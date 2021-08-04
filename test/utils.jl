using StableRNGs
using MLJ


function asymptotics(estimator, problem_fn)
    abs_mean_errors = []
    abs_var_errors = []
    for n in [100, 1000, 10000, 100000]
        abserrors_at_n = []
        for i in 1:10
            rng = StableRNG(i)
            T, W, y, ATE = problem_fn(rng; n=n)
            fitresult, _, _ = MLJ.fit(estimator, 0, T, W, y)
            push!(abserrors_at_n, abs(ATE-fitresult.estimate))
        end
        push!(abs_mean_errors, mean(abserrors_at_n))
        push!(abs_var_errors, var(abserrors_at_n))
    end
    return abs_mean_errors, abs_var_errors
end