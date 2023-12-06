function update_counts!(dict, key)
    if haskey(dict, key)
        dict[key] += 1
    else
        dict[key] = 1
    end
end

function nuisance_function_counts(estimands)
    η_counts = Dict()
    for Ψ in estimands
        for η in nuisance_functions_iterator(Ψ)
            update_counts!(η_counts, η)
        end
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
        # Append cache
        ηs = collect(nuisance_functions_iterator(Ψ))
        for η ∈ ηs
            if η ∉ cache
                compcost += 1
                push!(cache, η)
            end
        end
        # Update maxmem
        maxmem = max(maxmem, length(cache))
        verbosity > 0 && @info string("Cache size after estimand $estimand_index: ", length(cache))
        # Free cache from models that are not useful anymore
        for η in ηs
            η_counts[η] -= 1
            if η_counts[η] <= 0
                pop!(cache, η)
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
        candidate_min = n_uniques_nuisance_functions(Ψ)
        if candidate_min > min_maxmem_lowerbound
            min_maxmem_lowerbound = candidate_min
        end
    end
    return min_maxmem_lowerbound
end

estimands_permutation_generator(estimands) = Combinatorics.permutations(estimands)

function propensity_score_group_based_permutation_generator(groups)
    group_permutations = Combinatorics.permutations(collect(keys(groups)))
    return (vcat((groups[ps_key] for ps_key in group_perm)...) for group_perm in group_permutations)
end

"""
    brute_force_ordering(estimands; η_counts = nuisance_function_counts(estimands))

Finds an optimal ordering of the estimands to minimize maximum cache size. 
The approach is a brute force one, all permutations are generated and evaluated, 
if a minimum is found fast it is immediatly returned.
The theoretical complexity is in O(N!). However due to the stop fast approach and 
the shuffling, this is actually expected to be much smaller than that.
"""
function brute_force_ordering(estimands; 
    permutation_generator = estimands_permutation_generator(estimands), 
    η_counts=nuisance_function_counts(estimands), 
    do_shuffle=true, 
    rng=Random.default_rng(), 
    verbosity=0
    )
    estimands = do_shuffle ? shuffle(rng, estimands) : estimands
    optimal_ordering = estimands
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
    groupby_by_propensity_score(estimands)

Group parameters per propensity score and order each group by outcome_mean.
"""
function groupby_by_propensity_score(estimands)
    groups = Dict()
    for Ψ in estimands
        propensity_score_key_ = propensity_score_key(Ψ)
        outcome_mean_key_ = outcome_mean_key(Ψ)
        if haskey(groups, propensity_score_key_)
            propensity_score_group = groups[propensity_score_key_]
            if haskey(propensity_score_group, outcome_mean_key_)
                push!(propensity_score_group[outcome_mean_key_], Ψ)
            else
                propensity_score_group[outcome_mean_key_] = Any[Ψ]
            end
        else
            groups[propensity_score_key_] = Dict()
            groups[propensity_score_key_][outcome_mean_key_] = Any[Ψ]
        end
        
    end
    return Dict(key => vcat(values(groups[key])...) for key in keys(groups))
end

"""
    groups_ordering(estimands)

This will order estimands based on: propensity score first, outcome mean second. This heuristic should 
work reasonably well in practice. It could be optimized further by:
- Organising the propensity score groups that share similar components to be close together. 
- Brute forcing the ordering of these groups to find an optimal one.
"""
function groups_ordering(estimands; brute_force=false, do_shuffle=true, rng=Random.default_rng(), verbosity=0)
    # Group estimands based on propensity_score first and outcome_mean second
    groups = groupby_by_propensity_score(estimands)

    # Brute force on the propensity score groups
    if brute_force
        return brute_force_ordering(estimands; 
            permutation_generator = propensity_score_group_based_permutation_generator(groups), 
            do_shuffle=do_shuffle, 
            rng=rng, 
            verbosity=verbosity
        )
    else
        return vcat(values(groups)...)
    end
end