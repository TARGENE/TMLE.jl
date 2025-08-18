# TMLE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://targene.github.io/TMLE.jl/stable/)
![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/TARGENE/TMLE.jl/CI.yml?branch=main)
![Codecov](https://img.shields.io/codecov/c/github/TARGENE/TMLE.jl/main)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/TARGENE/TMLE.jl)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08446/status.svg)](https://doi.org/10.21105/joss.08446)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16884217.svg)](https://doi.org/10.5281/zenodo.16884217)

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
@article{Labayle_TMLE_jl_Targeted_Minimum_2025,
author = {Labayle, Olivier and Ponting, Chris P. and van der Laan, Mark J. and Khamseh, Ava and Beentjes, Sjoerd Viktor},
doi = {10.21105/joss.08446},
journal = {Journal of Open Source Software},
month = aug,
number = {112},
pages = {8446},
title = {{TMLE.jl: Targeted Minimum Loss-Based Estimation in Julia}},
url = {https://joss.theoj.org/papers/10.21105/joss.08446},
volume = {10},
year = {2025}
}
```

- APA

Labayle, O., Ponting, C. P., van der Laan, M. J., Khamseh, A., & Beentjes, S. V. (2025). TMLE.jl: Targeted Minimum Loss-Based Estimation in Julia. Journal of Open Source Software, 10(112), 8446. https://doi.org/10.21105/joss.08446

## Contact

A bug, a question or want to say hello? Please fill an [issue](https://github.com/TARGENE/TMLE.jl/issues).

