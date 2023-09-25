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
    @test parents(scm, :T₁) == [:W₁]
    @test parents(scm, :T₂) == [:W₂]
    @test parents(scm, :Y) == [:T₁, :T₂, :W₁, :W₂, :C]
    @test parents(scm, :C) == []
    # Another SCM
    scm = SCM(
        :Y₁ => [:T₁, :W₁],
        :T₁ => [:W₁],
        :Y₂ => [:T₁, :T₂, :W₁, :W₂, :C],
        :T₂ => [:W₂]
    )
    @test parents(scm, :Y₁) == [:T₁, :W₁]
    @test parents(scm, :T₁) == [:W₁]
    @test parents(scm, :Y₂) == [:T₁, :W₁, :T₂, :W₂, :C]
    @test parents(scm, :T₂) == [:W₂]
end

@testset "Test StaticSCM" begin
    scm = StaticSCM(
        outcomes = [:Y₁, :Y₂],
        treatments = [:T₁, :T₂, :T₃],
        confounders = [:W₁, :W₂],
        outcome_extra_covariates = [:C]
    )
    @test parents(scm, :Y₁) == parents(scm, :Y₂) == [:T₁, :T₂, :T₃, :W₁, :W₂, :C]
    @test parents(scm, :T₁) == parents(scm, :T₂) == parents(scm, :T₃) == [:W₁, :W₂]
end

end
true