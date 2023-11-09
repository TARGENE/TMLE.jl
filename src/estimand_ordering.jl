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

estimands_permutation_generator(estimands) = Combinatorics.permutations(estimands)

function get_propensity_score_groups(estimands_and_nuisances)
    ps_groups = [[]]
    current_ps = estimands_and_nuisances[1][2]
    for (index, Ψ_and_ηs) ∈ enumerate(estimands_and_nuisances)
        new_ps = Ψ_and_ηs[2]
        if new_ps == current_ps
            push!(ps_groups[end], index)
        else
            current_ps = new_ps
            push!(ps_groups, [index])
        end
    end
    return ps_groups
end

function propensity_score_group_based_permutation_generator(estimands, estimands_and_nuisances)
    ps_groups = get_propensity_score_groups(estimands_and_nuisances)
    group_permutations = Combinatorics.permutations(collect(1:length(ps_groups)))
    return (vcat((estimands[ps_groups[index]] for index in group_perm)...) for group_perm in group_permutations)
end

"""
    brute_force_ordering(estimands; η_counts = nuisance_counts(estimands))

Finds an optimal ordering of the estimands to minimize maximum cache size. 
The approach is a brute force one, all permutations are generated and evaluated, 
if a minimum is found fast it is immediatly returned.
The theoretical complexity is in O(N!). However due to the stop fast approach and 
the shuffling, this is actually expected to be much smaller than that.
"""
function brute_force_ordering(estimands; permutation_generator = estimands_permutation_generator(estimands), η_counts=nuisance_counts(estimands), do_shuffle=true, rng=Random.default_rng(), verbosity=0)
    optimal_ordering = estimands
    estimands = do_shuffle ? shuffle(rng, estimands) : estimands
    min_maxmem_lowerbound = get_min_maxmem_lowerbound(estimands)
    optimal_maxmem, _ = evaluate_proxy_costs(estimands, η_counts)
    for perm ∈ permutation_generator
        perm_maxmem, _ = evaluate_proxy_costs(perm, η_counts)
        if perm_maxmem < optimal_maxmem
            optimal_ordering = perm
            optimal_maxmem = perm_maxmem
        end
        # Stop fast if the lower bound is reached
        if optimal_maxmem == min_maxmem_lowerbound
            verbosity > 0 && @info(string("Lower bound reached, stopping."))
            return optimal_ordering
        end
    end
    return optimal_ordering
end

"""
    groups_ordering(estimands)

This will order estimands based on: propensity score first, outcome mean second. This heuristic should 
work reasonably well in practice. It could be optimized further by:
- Organising the propensity score groups that share similar components to be close together. 
- Brute forcing the ordering of these groups to find an optimal one.
"""
function groups_ordering(estimands; brute_force=false, do_shuffle=true, rng=Random.default_rng(), verbosity=0)
    # Sort estimands based on propensity_score first and outcome_mean second
    estimands_and_nuisances = []
    for Ψ in estimands
        η = TMLE.get_relevant_factors(Ψ)
        push!(estimands_and_nuisances, (Ψ, η.propensity_score, η.outcome_mean))
    end
    sort!(estimands_and_nuisances, by = x -> (Tuple(TMLE.variables(ps) for ps in x[2]), TMLE.variables(x[3])))
    
    # Sorted estimands only
    estimands = [x[1] for x in estimands_and_nuisances]

    # Brute force on the propensity score groups
    if brute_force
        return brute_force_ordering(estimands; 
            permutation_generator = propensity_score_group_based_permutation_generator(estimands, estimands_and_nuisances), 
            do_shuffle=do_shuffle, 
            rng=rng, 
            verbosity=verbosity
        )
    else
        return estimands
    end
end