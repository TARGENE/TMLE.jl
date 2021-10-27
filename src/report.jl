mutable struct Report <: MLJ.Model end

function MLJ.fit(model::Report, 
                 verbosity::Int, 
                 ct_fluct,
                 observed_fluct, 
                 covariate,
                 y)

    estimate = mean(ct_fluct)
    
    inf_curve = covariate .* (float(y) .- observed_fluct) .+ ct_fluct .- estimate

    fitresult = (
        estimate=estimate, 
        stderror=sqrt(var(inf_curve)/nrows(y)), 
        mean_inf_curve=mean(inf_curve)
        )
    return fitresult, nothing, nothing
end


function predict(model::Report, fitresult, Xnew)
    return Node()
end