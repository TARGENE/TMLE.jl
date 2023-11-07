function maybe_update_counts!(dict, key)
    if haskey(dict, key)
        dict[key] += 1
    else
        dict[key] = 1
    end
end

function nuisance_counts(estimands)
    η_counts = Dict()
    for Ψ in estimands
        η = TMLE.get_relevant_factors(Ψ)
        for ps in η.propensity_score
            maybe_update_counts!(η_counts, ps)
        end
        maybe_update_counts!(η_counts, η.outcome_mean)
    end
    return η_counts
end

"""
    evaluate_proxy_costs(estimands, η_counts; verbosity=0)

This function evaluates proxy measures for both:
    - The computational cost (1 per model fit)
    - The maximum memory used (1 per model)

that would be used while performing estimation for an ordered list of estimands. 
This is assuming that after each estimation, we purge the cache from models that will 
not be subsequently used. That way, we are guaranteed that we minimize computational cost.
"""
function evaluate_proxy_costs(estimands, η_counts; verbosity=0)
    η_counts = deepcopy(η_counts)
    cache = Set()
    maxmem = 0
    compcost = 0
    for (estimand_index, Ψ) ∈ enumerate(estimands)
        η = get_relevant_factors(Ψ)
        models = (η.propensity_score..., η.outcome_mean)
        # Append cache
        for model in models
            if model ∉ cache
                compcost += 1
                push!(cache, model)
            end
        end
        # Update maxmem
        maxmem = max(maxmem, length(cache))
        verbosity > 0 && @info string("Cache size after estimand $estimand_index: ", length(cache))
        # Free cache from models that are not useful anymore
        for model in models
            η_counts[model] -= 1
            if η_counts[model] <= 0
                pop!(cache, model)
            end
        end
    end
    
    return maxmem, compcost
end

"""
    get_min_maxmem_lowerbound(estimands)

The maximum number of models for a single estimand is a lower bound 
on the cache size. It can be computed in a single pass, i.e. in O(N).
"""
function get_min_maxmem_lowerbound(estimands)
    min_maxmem_lowerbound = 0
    for Ψ in estimands
        η = get_relevant_factors(Ψ)
        candidate_min = length((η.propensity_score..., η.outcome_mean))
        if candidate_min > min_maxmem_lowerbound
            min_maxmem_lowerbound = candidate_min
        end
    end
    return min_maxmem_lowerbound
end

"""
    brute_force_ordering(estimands; η_counts = nuisance_counts(estimands))

Finds an optimal ordering of the estimands to minimize maximum cache size. 
The approach is a brute force one, all permutations are generated and evaluated, 
if a minimum is found fast it is immediatly returned.
The theoretical complexity is in O(N!). However due to the stop fast approach and 
the shuffling, this is actually expected to be much smaller than that.
"""
function brute_force_ordering(estimands; η_counts=nuisance_counts(estimands), do_shuffle=true, rng=Random.default_rng(), verbosity=0)
    optimal_ordering = estimands
    estimands = do_shuffle ? shuffle(rng, estimands) : estimands
    min_maxmem_lowerbound = get_min_maxmem_lowerbound(estimands)
    optimal_maxmem, optimal_compcost = evaluate_proxy_costs(estimands, η_counts)
    for perm ∈ Combinatorics.permutations(estimands)
        perm_maxmem, _ = evaluate_proxy_costs(perm, η_counts)
        if perm_maxmem < optimal_maxmem
            optimal_ordering = perm
            optimal_maxmem = perm_maxmem
        end
        # Stop fast if the lower bound is reached
        if optimal_maxmem == min_maxmem_lowerbound
            verbosity > 0 && @info(string("Lower bound reached, stopping."))
            return optimal_ordering, optimal_maxmem, optimal_compcost
        end
    end
    return optimal_ordering, optimal_maxmem, optimal_compcost
end