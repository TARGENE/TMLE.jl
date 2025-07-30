"""
    counterfactual_aggregate(Ψ::StatisticalCMCompositeEstimand, Q::Machine)

This is a counterfactual aggregate that depends on the parameter of interest.

For the ATE with binary treatment, confounded by W it is equal to:

``ctf_agg = Q(1, W) - Q(0, W)``

"""
function counterfactual_aggregate(Ψ::StatisticalCMCompositeEstimand, Q, dataset)
    X = selectcols(dataset, Q.estimand.parents)
    Ttemplate = selectcols(X, treatments(Ψ))
    n = nrows(Ttemplate)
    ctf_agg = zeros(n)
    # Loop over Treatment settings
    for (vals, sign) in indicator_fns(Ψ)
        # Counterfactual dataset for a given treatment setting
        T_ct = counterfactualTreatment(vals, Ttemplate)
        X_ct = DataFrame((;(Symbol(colname) => colname ∈ names(T_ct) ? T_ct[!, colname] : X[!, colname] for colname in names(X))...))
        # Counterfactual mean
        ctf_agg .+= sign .* expected_value(Q, X_ct)
    end
    return ctf_agg
end

plugin_estimate(ctf_aggregate) = mean(ctf_aggregate)

function ccw_plugin_estimate(
    Ψ::StatisticalCMCompositeEstimand,
    targeted::MLCMRelevantFactors,
    nomiss::DataFrame
    )
    # targeted.outcome_mean  :: ConditionalDistributionEstimate for Q*_n
    # targeted.marginal_w    :: WeightedEmpiricalDistributionEstimate for Q_{W,n}^0
    Qstar = targeted.outcome_mean
    QW    = targeted.marginal_w

    Wvals = QW[1].values
    Wwts  = QW[1].weights
    n_atoms = length(Wvals)


    parents = Qstar.estimand.parents  
    df1 = DataFrame()
    df0 = DataFrame()

    for p in parents
        orig_col = nomiss[!, p] # a Vector{T} of the original column
        # make a new empty vector of length n_atoms, same element-type
        newcol1 = similar(orig_col, n_atoms)
        newcol0 = similar(orig_col, n_atoms)           
        if p == collect(keys(Ψ.treatment_values))[1] 
            # get the original levels out of nomiss
            if orig_col isa CategoricalVector
                # for Categorical, preserve levels & ordering
                l = CategoricalArrays.levels(orig_col)
                ordered = isordered(orig_col)
                # build new categorical arrays
                case_val    = Ψ.treatment_values[p].case
                control_val = Ψ.treatment_values[p].control
                newcol1 = categorical(fill(case_val, n_atoms); levels=l, ordered=ordered)
                newcol0 = categorical(fill(control_val, n_atoms); levels=l, ordered=ordered)
            else
                # numeric or Bool
                newcol1 .= Ψ.treatment_values[p].case
                newcol0 .= Ψ.treatment_values[p].control
            end
        elseif p == QW[1].estimand.variable  # the W-column
            newcol1 .= Wvals
            newcol0 .= Wvals
        else
            error("Unexpected covariate $(p) in ccw_plugin_estimate")
        end

        df1[!, p] = newcol1
        df0[!, p] = newcol0
    end

    # predict
    ŷ1 = expected_value(predict(Qstar, df1))
    ŷ0 = expected_value(predict(Qstar, df0))

    return sum(Wwts .* (ŷ1 .- ŷ0))
end

function ccw_IC(IC, dataset, relevant_factors, q0)
    y = dataset[!, relevant_factors.outcome_mean.outcome]
    is_case = y .== true 
    case_idxs = findall(is_case)
    control_idxs = findall(!, is_case)

    n_case = length(case_idxs)
    n_ctrl = length(control_idxs)
    @assert n_case > 0 "No cases found in dataset"
    @assert n_ctrl > 0 "No controls found in dataset"

    J = n_ctrl / n_case
    g = floor(Int, J)
    r = n_ctrl - g * n_case
    sizes = [ i ≤ r ? g+1 : g for i in 1:n_case ]

    IC_ccw = Vector{Float64}(undef, n_case)
    pos = 1
    for (i, ci) in enumerate(case_idxs)
        s = sizes[i]
        group = control_idxs[pos:pos+s-1]
        sum_ctrl = sum(IC[group])
        IC_ccw[i] = q0 * IC[ci] + (1 - q0) * (sum_ctrl / J)
        pos += s
    end

    return IC_ccw

end


"""
    ∇W(ctf_agg, Ψ̂)

Computes the projection of the gradient on the (W) space.
"""
∇W(ctf_agg, Ψ̂) = ctf_agg .- Ψ̂

"""
    ∇YX(H, y, Ey, w)

Computes the projection of the gradient on the (Y | X) space.

- H: Clever covariate
- y: Outcome
- Ey: Expected value of the outcome
- w: Weights
"""
∇YX(H, y, Ey, w) = H .* w .* (y .- Ey)

"""
    gradient_Y_X(cache)

∇_YX(w, t, c) = covariate(w, t)  ̇ (y - E[Y|w, t, c])

This part of the gradient is evaluated on the original dataset. All quantities have been precomputed and cached.
"""
function ∇YX(Ψ::StatisticalCMCompositeEstimand, Q, G, dataset; ps_lowerbound=1e-8)
    # Maybe can cache some results (H and E[Y|X]) to improve perf here
    H, w = clever_covariate_and_weights(Ψ, G, dataset; ps_lowerbound=ps_lowerbound)
    y = float(dataset[!, Q.estimand.outcome])
    Ey = expected_value(Q, dataset)
    return ∇YX(H, y, Ey, w)
end


function gradient_and_plugin_estimate(Ψ::StatisticalCMCompositeEstimand, factors, dataset; ps_lowerbound=1e-8)
    Q = factors.outcome_mean
    G = factors.propensity_score
    ctf_agg = counterfactual_aggregate(Ψ, Q, dataset)
    Ψ̂ = plugin_estimate(ctf_agg)
    IC = ∇YX(Ψ, Q, G, dataset; ps_lowerbound = ps_lowerbound) .+ ∇W(ctf_agg, Ψ̂)
    return IC, Ψ̂
end

train_validation_indices_from_ps(::MLConditionalDistribution) = nothing
train_validation_indices_from_ps(factor::SampleSplitMLConditionalDistribution) = factor.train_validation_indices

train_validation_indices_from_factors(factors) = 
    train_validation_indices_from_ps(first(factors.propensity_score))