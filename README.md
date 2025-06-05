# TMLE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://targene.github.io/TMLE.jl/stable/)
![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/TARGENE/TMLE.jl/CI.yml?branch=main)
![Codecov](https://img.shields.io/codecov/c/github/TARGENE/TMLE.jl/main)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/TARGENE/TMLE.jl)

Causal inference is essential for understanding the effect of interventions in real-world settings, such as determining whether a treatment improves health outcomes or whether a gene variant contributes to disease risk. Traditional statistical methods, such as linear regression or propensity score matching, often rely on strong modeling assumptions and may fail to provide valid inference when these assumptions are violated—particularly in the presence of high-dimensional data or model misspecification.

**TMLE.jl** is a Julia package that implements [Targeted Maximum Likelihood Estimation](https://link.springer.com/book/10.1007/978-1-4419-9782-1) (TMLE), a general framework for causal effect estimation that combines machine learning with principles from semiparametric statistics. TMLE provides doubly robust, efficient, and flexible estimation of causal parameters in observational and experimental studies.

## Installation

TMLE.jl can be installed via the Package Manager and supports Julia `v1.10` and greater.

```Pkg
Pkg> add TMLE
```

## Documentation

For more information, please visit the [documentation](https://targene.github.io/TMLE.jl/stable/)

## Citation

If you use TMLE.jl for your own work and would like to cite us, here are the BibTeX and APA formats:

- BibTeX

```bibtex
@software{Labayle_TMLE_jl,
    author = {Labayle, Olivier and Khamseh, Ava and Ponting, Chris and Beentjes, Sjoerd},
    title = {{TMLE.jl}},
    url = {https://github.com/TARGENE/TMLE.jl}
}
```

- APA

Labayle, O., Khamseh, A., & Ponting, C., Beentjes, S. TMLE.jl [Computer software]. https://github.com/TARGENE/TMLE.jl

## Contact

A bug, a question or want to say hello? Please fill an [issue](https://github.com/TARGENE/TMLE.jl/issues).

