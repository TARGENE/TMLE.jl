using TMLE
using Documenter

DocMeta.setdocmeta!(TMLE, :DocTestSetup, :(using TMLE); recursive=true)

makedocs(;
    modules=[TMLE],
    authors="Olivier Labayle",
    repo="https://github.com/olivierlabayle/TMLE.jl/blob/{commit}{path}#{line}",
    sitename="TMLE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://olivierlabayle.github.io/TMLE.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "User Guide" => "user_guide.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/olivierlabayle/TMLE.jl",
    devbranch="main",
    push_preview=true
)
