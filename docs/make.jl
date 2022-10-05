using TMLE
using Documenter
using Literate

DocMeta.setdocmeta!(TMLE, :DocTestSetup, :(using TMLE); recursive=true)

# Generate Literate markdown pages
@info "Building Literate pages..."
examples_dir = joinpath(@__DIR__, "../examples")
build_examples_dir =  joinpath(@__DIR__, "src", "examples/")
for file in readdir(examples_dir)
    Literate.markdown(joinpath(examples_dir, file), build_examples_dir;documenter=true)
end

@info "Running makedocs..."
makedocs(;
    modules=[TMLE],
    authors="Olivier Labayle",
    repo="https://github.com/olivierlabayle/TMLE.jl/blob/{commit}{path}#{line}",
    sitename="TMLE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://olivierlabayle.github.io/TMLE.jl",
        assets=String["assets/logo.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "User Guide" => "user_guide.md",
        "Examples" => [
            joinpath("examples", "introduction_to_targeted_learning.md"), 
            joinpath("examples", "super_learning.md")
            ],
        "API Reference" => "api.md"
    ],
)

@info "Deploying docs..."
deploydocs(;
    repo="github.com/olivierlabayle/TMLE.jl",
    devbranch="main",
    push_preview=true
)
