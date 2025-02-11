using TMLE
using Documenter
using Literate

DocMeta.setdocmeta!(TMLE, :DocTestSetup, :(using TMLE); recursive=true)

## Generate Literate markdown pages
@info "Building Literate pages..."
examples_dir = joinpath(@__DIR__, "../examples")
build_examples_dir =  joinpath(@__DIR__, "src", "examples/")
for file in readdir(examples_dir)
    if endswith(file, "jl")
        Literate.markdown(joinpath(examples_dir, file), build_examples_dir;documenter=true)
    end
end

@info "Running makedocs..."
makedocs(;
    modules=[TMLE],
    authors="Olivier Labayle",
    repo="https://github.com/TARGENE/TMLE.jl/blob/{commit}{path}#{line}",
    sitename="TMLE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TARGENE.github.io/TMLE.jl",
        assets=String["assets/logo.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Walk Through" => "walk_through.md",
        "User Guide" => [joinpath("user_guide", f) for f in 
            ("scm.md", "estimands.md", "estimation.md")],
        "Examples" => [
            joinpath("examples", "super_learning.md"),
            joinpath("examples", "double_robustness.md")
            ],
        "Integrations" => "integrations.md",
        "Estimators' Cheat Sheet" => "estimators_cheatsheet.md",
        "Learning Resources" => "resources.md",
        "API Reference" => "api.md",
        
    ],
    pagesonly=true,
    clean = true,
    checkdocs=:exports
)

@info "Deploying docs..."
deploydocs(;
    repo="github.com/TARGENE/TMLE.jl",
    devbranch="main",
    push_preview=true
)
