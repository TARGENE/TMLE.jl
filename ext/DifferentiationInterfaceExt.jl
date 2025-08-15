module DifferentiationInterfaceExt

using TMLE
using DifferentiationInterface

function TMLE.compose(f, Ψ̂::TMLE.JointEstimate, backend)
    point_estimate = estimate(Ψ̂)
    Σ = Ψ̂.cov
    f₀, J = value_and_jacobian(f, backend, point_estimate)
    σ₀ = J * Σ * J'
    estimand = TMLE.ComposedEstimand(f, Ψ̂.estimand)
    return TMLE.ComposedEstimate(estimand, f₀, σ₀, Ψ̂.n)
end

end