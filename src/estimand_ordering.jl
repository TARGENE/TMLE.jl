
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


function brute_force_ordering(estimands; η_counts = nuisance_counts(estimands))
    optimal_ordering = estimands
    optimal_maxmem, optimal_compcost = evaluate_proxy_costs(estimands, η_counts)
    for perm ∈ Combinatorics.permutations(estimands)
        perm_maxmem, _ = evaluate_proxy_costs(perm, η_counts)
        if perm_maxmem < optimal_maxmem
            optimal_ordering = perm
            optimal_maxmem = perm_maxmem
        end
    end
    return optimal_ordering, optimal_maxmem, optimal_compcost
end