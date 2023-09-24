# TMLE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://targene.github.io/TMLE.jl/stable/)
![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/TARGENE/TMLE.jl/CI.yml?branch=main)
![Codecov](https://img.shields.io/codecov/c/github/TARGENE/TMLE.jl/main)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/TARGENE/TMLE.jl)

This package enables the estimation of various Causal Inference related estimands using Targeted Minimum Loss-Based Estimation (TMLE). TMLE.jl is based on top of [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/), which means any MLJ compliant machine-learning model can be used here.

**New to TMLE.jl?** [Get started now](https://targene.github.io/TMLE.jl/stable/)

## Notes to self

- For multiple TMLE steps, convergence when the mean of the IC is below: σ/n where σ is the standard deviation of the IC.
- Try make selectcols preserve input type (see MLJModel has a facility)
- Watch for discrepancy in train_validation_indices with missing data...
- Check that for IATEs simulations, the true effect size is actually an additive interaction and not a multiplicative one.
- clean imports in test files
- clean export list
- look into dataset management and missing values
- See about ordering of parameters