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



end

true