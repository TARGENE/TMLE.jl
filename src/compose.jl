
function compose(f, tmleresports...)
    Σ = multivariateCovariance((tlr.eic for tlr in tmleresports)...)
    ∇f = collect(gradient(f, (tlr.estimate for tlr in tmleresports)...))
    return ∇f'*Σ*∇f
end

function multivariateCovariance(args...)
    D = hcat(args...)
    n = size(D, 1)
    return D'D/n
end
