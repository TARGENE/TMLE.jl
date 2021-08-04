using Test



@time begin
    @test include("ate.jl")
    @test include("interaction_ate.jl")
end