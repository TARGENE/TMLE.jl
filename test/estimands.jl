module TestEstimands

using TMLE
using Test
using DataFrames

@testset "Test ConditionalDistribution" begin
    distr = TMLE.ConditionalDistribution("Y", ["C", :A])
    @test distr.outcome === :Y
    @test distr.parents === (:A, :C)
    @test TMLE.variables(distr) == (:Y, :A, :C)
    dataset = DataFrame(Y=randn(100), C=rand(100), A=rand(100))
    X, y = TMLE.get_mlj_model_inputs(distr, dataset)
    @test X == dataset[!, [:A, :C]]
    @test y == dataset[!, :Y]
end

@testset "Test RieszRepresenter" begin
    Ψ = TMLE.ATE(
        outcome=:Y,
        treatment_values=(T_1=(case=1, control=0), T_2=(case=1, control=0)),
        treatment_confounders=(T_1=[:W₁, :W₂], T_2=[:A])
    )
    riesz_representer = TMLE.RieszRepresenter(Ψ)
    @test riesz_representer.Ψ === Ψ
    @test TMLE.variables(riesz_representer) == [:T_1, :T_2, :A, :W₁, :W₂]
    
    dataset = DataFrame(
        T_1=rand(100), 
        T_2=rand(100), 
        A=rand(100), 
        W₁=rand(100), 
        W₂=rand(100),
        Y=randn(100)
    )
    X = TMLE.get_mlj_model_inputs(riesz_representer, dataset)
    @test X == dataset[!, [:T_1, :T_2, :A, :W₁, :W₂]]
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

    # ComposedEstimand
    Main.eval(:(difference(x, y) = x - y))
    composed = ComposedEstimand(Main.difference, joint)
    composed_dict = TMLE.to_dict(composed)
    composed_from_dict = TMLE.from_dict!(composed_dict)
    @test composed_from_dict == composed

    # Anonymous function will raise
    diff = ComposedEstimand((x,y) -> x - y, joint)
    msg = "The function of a ComposedEstimand cannot be anonymous to be converted to a dictionary."
    @test_throws ArgumentError(msg) TMLE.to_dict(diff)
end


end

true