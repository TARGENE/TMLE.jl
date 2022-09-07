using TMLE

abserrors(results, Ψ₀) = [abs(TMLE.estimate(r) - Ψ₀) for r in results]

all_tmle_better_than_initial(tmle, initial, Ψ₀) = all(abserrors(tmle, Ψ₀) .<= abserrors(initial, Ψ₀))
first_better_than_last(results, Ψ₀) = 
    abs((TMLE.estimate(results[end]) - Ψ₀) / Ψ₀) < abs((TMLE.estimate(results[1]) - Ψ₀) / Ψ₀)

tolerance(res, Ψ₀, tol) = ((TMLE.estimate(res) - Ψ₀) / Ψ₀) < tol

all_solves_ice(tmle_results; tol=1e-10) = all(mean(r.IC) < tol for r in tmle_results)

function asymptotics(Ψ, η_spec, problem_fn, rng, Ns)
    tmle_results = []
    initial_results = []
    Ψ₀ = 0
    for n in Ns
        dataset, Ψ₀ = problem_fn(rng; n=n)
        tmle_result, initial_result, cache = tmle(Ψ, η_spec, dataset; verbosity=0)
        push!(tmle_results, tmle_result)
        push!(initial_results, initial_result)
    end
    return tmle_results, initial_results, Ψ₀
end