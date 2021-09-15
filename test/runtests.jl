using Test


@time begin
    println("WARNING: ATE tests are bypassed for now")
    # @test include("ate.jl")
    @test include("utils.jl")
    @test include("interaction_ate.jl")

end