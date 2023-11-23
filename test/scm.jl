module TestSCM

using Test
using TMLE

@testset "Test SCM" begin
    # An empty SCM
    scm = SCM()
    add_equations!(
        scm,
        :Y => [:T₁, :T₂, :W₁, :W₂, :C],
        :T₁ => [:W₁]
    )
    add_equation!(scm, :T₂ => [:W₂])
    @test parents(scm, :T₁) == Set([:W₁])
    @test parents(scm, :T₂) == Set([:W₂])
    @test parents(scm, :Y) == Set([:T₁, :T₂, :W₁, :W₂, :C])
    @test parents(scm, :C) == Set([])
    expected_vertices = Set([:Y, :T₁, :T₂, :W₁, :W₂, :C])
    @test Set(vertices(scm)) == expected_vertices

    scm_dict = TMLE.to_dict(scm)
    @test scm_dict == Dict(
        :type => "SCM", 
        :equations => [
            Dict(:parents => Any[], :outcome => :C), 
            Dict(:parents => Any[], :outcome => :W₁), 
            Dict(:parents => [:C, :W₁, :T₁, :W₂, :T₂], :outcome => :Y), 
            Dict(:parents => [:W₁], :outcome => :T₁), 
            Dict(:parents => Any[], :outcome => :W₂), 
            Dict(:parents => [:W₂], :outcome => :T₂)
        ]
    )
    reconstructed_scm = TMLE.from_dict!(scm_dict)
    for vertexlabel ∈ expected_vertices
        @test parents(reconstructed_scm, vertexlabel) == parents(scm, vertexlabel)
    end

    # Another SCM
    scm = SCM([
        :Y₁ => [:T₁, :W₁],
        :T₁ => [:W₁],
        :Y₂ => [:T₁, :T₂, :W₁, :W₂, :C],
        :T₂ => [:W₂]
    ]
    )
    @test parents(scm, :Y₁) == Set([:T₁, :W₁])
    @test parents(scm, :T₁) == Set([:W₁])
    @test parents(scm, :Y₂) == Set([:T₁, :W₁, :T₂, :W₂, :C])
    @test parents(scm, :T₂) == Set([:W₂])
    @test Set(vertices(scm)) == Set([:Y₁, :Y₂, :T₁, :T₂, :W₁, :W₂, :C])

end

@testset "Test StaticSCM" begin
    scm = StaticSCM(
        outcomes = [:Y₁, :Y₂],
        treatments = [:T₁, :T₂, :T₃],
        confounders = [:W₁, :W₂],
    )
    @test parents(scm, :Y₁) == parents(scm, :Y₂) == Set([:T₁, :T₂, :T₃, :W₁, :W₂])
    @test parents(scm, :T₁) == parents(scm, :T₂) == parents(scm, :T₃) == Set([:W₁, :W₂])

    reconstructed_scm = TMLE.from_dict!(Dict(
        :type => "StaticSCM",
        :outcomes => [:Y₁, :Y₂],
        :treatments => [:T₁, :T₂, :T₃],
        :confounders => [:W₁, :W₂]
    ))
    for vertexlabel ∈ [:Y₁, :Y₂, :T₁, :T₂, :T₃, :W₁, :W₂]
        @test parents(reconstructed_scm, vertexlabel) == parents(scm, vertexlabel)
    end

end

end
true