module TestUtils

using Test
using TMLE

@testset "Test interaction_combinations" begin
    query = (t₁=["a", "b"], t₂ = ["c", "d"], t₃ = [true, false])
    combinations = TMLE.interaction_combinations(query)
    expected_comb = [(t₁ = "a", t₂ = "c", t₃ = true),
                     (t₁ = "b", t₂ = "c", t₃ = true),
                     (t₁ = "a", t₂ = "d", t₃ = true),
                     (t₁ = "b", t₂ = "d", t₃ = true),
                     (t₁ = "a", t₂ = "c", t₃ = false),
                     (t₁ = "b", t₂ = "c", t₃ = false),
                     (t₁ = "a", t₂ = "d", t₃ = false),
                     (t₁ = "b", t₂ = "d", t₃ = false)]
    for (i, comb) in enumerate(combinations)
        @test comb == expected_comb[i]
    end
end

@testset "Test indicator_fns" begin
    # For 2 treatment variables:
    # The "case" treatment is: (true, false) 
    query = (t₁=[true, false], t₂ = [false, true])
    indicators = TMLE.indicator_fns(query)
    @test indicators == Dict(
        (t₁ = 1, t₂ = 0) => 1,
        (t₁ = 0, t₂ = 0) => -1,
        (t₁ = 1, t₂ = 1) => -1,
        (t₁ = 0, t₂ = 1) => 1
    )

    # For 3 treatment variables:
    # The "case" treatment is: (a, c, true) 
    query = (t₁=["a", "b"], t₂ = ["c", "d"], t₃ = [true, false])
    indicators = TMLE.indicator_fns(query)
    @test indicators == Dict(
        (t₁ = "b", t₂ = "c", t₃ = 1) => -1, # (-1)^{2}=1
        (t₁ = "a", t₂ = "c", t₃ = 1) => 1,  # (-1)^{0}=1
        (t₁ = "b", t₂ = "d", t₃ = 0) => -1, # (-1)^{3}=-1
        (t₁ = "b", t₂ = "c", t₃ = 0) => 1,  # (-1)^{2}=1
        (t₁ = "a", t₂ = "d", t₃ = 1) => -1, # (-1)^{1}=-1
        (t₁ = "a", t₂ = "c", t₃ = 0) => -1, # (-1)^{1}=-1
        (t₁ = "a", t₂ = "d", t₃ = 0) => 1,  # (-1)^{2}=1
        (t₁ = "b", t₂ = "d", t₃ = 1) => 1   # (-1)^{2}=1
    )
end

end;

true