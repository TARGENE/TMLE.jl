module TestEstimands

using TMLE
using Test
using DataFrames

@testset "Test ConditionalDistribution" begin
    distr = TMLE.ConditionalDistribution("Y", ["C", 1, :A, ])
    @test distr.outcome === :Y
    @test distr.parents === (Symbol("1"), :A, :C)
    @test TMLE.variables(distr) == (:Y, Symbol("1"), :A, :C)
    # Test string representation
    @test TMLE.string_repr(distr) == "P₀(Y | 1, A, C)"
end

@testset "Test CMRelevantFactors" begin
    η = TMLE.CMRelevantFactors(
        outcome_mean=TMLE.ExpectedValue(:Y, [:T, :W]),
        propensity_score=TMLE.ConditionalDistribution(:T, [:W])
    )
    @test TMLE.variables(η) == (:Y, :T, :W)

    η = TMLE.CMRelevantFactors(
        outcome_mean=TMLE.ExpectedValue(:Y, [:T, :W]),
        propensity_score=(
            TMLE.ConditionalDistribution(:T₁, [:W₁]),
            TMLE.ConditionalDistribution(:T₂, [:W₂₁, :W₂₂])
        )
    )
    @test TMLE.variables(η) == (:Y, :T, :W, :T₁, :W₁, :T₂, :W₂₁, :W₂₂)
    # Test string representation
    @test TMLE.string_repr(η) == "Relevant Factors: \n- P₀(Y | T, W)\n- P₀(T₁ | W₁)\n- P₀(T₂ | W₂₁, W₂₂)"
end

@testset "Test JointEstimand and ComposedEstimand" begin
    ATE₁ = ATE(
        outcome=:Y,
        treatment_values = (T₁=(case=1, control=0), T₂=(case=1, control=0)),
        treatment_confounders = (T₁=[:W], T₂=[:W])
    )
    ATE₂ = ATE(
        outcome=:Y,
        treatment_values = (T₁=(case=2, control=1), T₂=(case=2, control=1)),
        treatment_confounders = (T₁=[:W], T₂=[:W])
    )
    # JointEstimand
    joint = JointEstimand(ATE₁, ATE₂)

    @test TMLE.propensity_score_key(joint) == ((:T₁, :T₂, :W), (:T₂, :W))
    @test TMLE.outcome_mean_key(joint) == ((:Y, :T₁, :T₂, :W),)

    joint_dict = TMLE.to_dict(joint)
    joint_from_dict = TMLE.from_dict!(joint_dict)
    @test joint_from_dict == joint

    # Test string representation
    @test TMLE.string_repr(joint) == "Joint Estimand:\n--------------\n- StatisticalATE\n\t- Outcome: Y\n\t- Treatment: T₁ => (control = 0, case = 1) & T₂ => (control = 0, case = 1)\n- StatisticalATE\n\t- Outcome: Y\n\t- Treatment: T₁ => (control = 1, case = 2) & T₂ => (control = 1, case = 2)"

    # ComposedEstimand
    Main.eval(:(difference(x, y) = x - y))
    composed = ComposedEstimand(Main.difference, joint)
    composed_dict = TMLE.to_dict(composed)
    composed_from_dict = TMLE.from_dict!(composed_dict)
    @test composed_from_dict == composed

    # Test string representation
    @test TMLE.string_repr(composed) == "Composed Estimand applying function `difference` to :\n----------------------------------------------------\n- StatisticalATE\n\t- Outcome: Y\n\t- Treatment: T₁ => (control = 0, case = 1) & T₂ => (control = 0, case = 1)\n- StatisticalATE\n\t- Outcome: Y\n\t- Treatment: T₁ => (control = 1, case = 2) & T₂ => (control = 1, case = 2)"

    # Anonymous function will raise
    diff = ComposedEstimand((x,y) -> x - y, joint)
    msg = "The function of a ComposedEstimand cannot be anonymous to be converted to a dictionary."
    @test_throws ArgumentError(msg) TMLE.to_dict(diff)
end


end

true