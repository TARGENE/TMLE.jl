module TestQuery

using TMLE
using Test
using CategoricalArrays


@testset "Test constructors" begin
    # Base constructor
    query = Query("my_query", (t₁=true, t₂=true), (t₁=false, t₂=false))
    @test query isa Query{NamedTuple{(:t₁, :t₂), Tuple{Bool, Bool}}}
    # Optional name constructor
    query = Query((t₁="CC", t₂="CG"), (t₁="GG", t₂="GG"))
    @test query isa Query{NamedTuple{(:t₁, :t₂), Tuple{String, String}}}
    @test query.name === nothing

    query = Query((t₁="CC", t₂="CG"), (t₁="GG", t₂="GG"), name="myquery")
    @test query.name == "myquery"

    # Only keyword constructor
    @test Query() isa Query
    query = Query(control=(t₁="GG", t₂="GG"), case=(t₁="CC", t₂="CG"))
    @test query.case == (t₁="CC", t₂="CG")
    @test query.control == (t₁="GG", t₂="GG")

end

@testset "Test Queries Misc" begin
    query = Query((t₁="CC", t₂="CG"), (t₁="GG", t₂="GG"))
    @test TMLE.variables(query) == (:t₁, :t₂)

    T = (
        t₁=categorical(["CC", "GG", "CC"]),
        t₂=categorical(["CC", "GG", "CC"]),
        t₃=categorical(["CC", "GG", "CC"]),
        )
    @test_throws ArgumentError TMLE.check_ordering((query,), T)
end

@testset "Test interaction_combinations" begin
    # With 1 treatment variable
    query = Query((t₁="a",), (t₁="b",))
    combinations = TMLE.interaction_combinations(query)
    expected_comb = [("a",),
                     ("b",)]
    for (i, comb) in enumerate(combinations)
        @test comb == expected_comb[i]
    end

    # With 3 treatment variables
    query = Query((t₁="a", t₂="c", t₃=true), (t₁="b", t₂="d", t₃=false))
    combinations = TMLE.interaction_combinations(query)
    expected_comb = [("a", "c", true),
                     ("b", "c", true),
                     ("a", "d", true),
                     ("b", "d", true),
                     ("a", "c", false),
                     ("b", "c", false),
                     ("a", "d", false),
                     ("b", "d", false)]
    for (i, comb) in enumerate(combinations)
        @test comb == expected_comb[i]
    end
end

@testset "Test indicator_fns" begin
    # For 1 treatment variable
    query = Query((t₁="a",),  (t₁="b",))
    indicators = TMLE.indicator_fns(query)
    @test indicators isa Base.ImmutableDict
    @test indicators[("a",)] == 1
    @test indicators[("b",)] == -1

    # For 2 treatment variables:
    # The "case" treatment is: (true, false) 
    query = Query((t₁=true, t₂ = false), (t₁=false, t₂ = true))
    indicators = TMLE.indicator_fns(query)

    @test indicators[(1, 0)] == 1
    @test indicators[(0, 0)] == -1
    @test indicators[(1, 1)] == -1
    @test indicators[(0, 1)] == 1


    # For 3 treatment variables:
    # The "case" treatment is: (a, c, true) 
    query = Query(case=(t₁="a", t₂="c", t₃=true), control=(t₁="b", t₂="d", t₃=false))
    indicators = TMLE.indicator_fns(query)
    @test indicators[("b", "c", 1)] == -1 # (-1)^{2}=1
    @test indicators[("a", "c", 1)] == 1  # (-1)^{0}=1
    @test indicators[("b", "d", 0)] == -1 # (-1)^{3}=-1
    @test indicators[("b", "c", 0)] == 1  # (-1)^{2}=1
    @test indicators[("a", "d", 1)] == -1 # (-1)^{1}=-1
    @test indicators[("a", "c", 0)] == -1 # (-1)^{1}=-1
    @test indicators[("a", "d", 0)] == 1  # (-1)^{2}=1
    @test indicators[("b", "d", 1)] == 1  # (-1)^{2}=1

end

end

true