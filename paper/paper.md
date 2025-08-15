---
title: 'TMLE.jl: Targeted Minimum Loss-Based Estimation In Julia.'
tags:
  - julia
  - statistics
  - semiparametric statistics
  - machine learning
  - causal inference
authors:
  - name: Olivier Labayle
    orcid: 0000-0002-3708-3706
    affiliation: "1"
  - name: Chris P. Ponting
    affiliation: "2"
    orcid: 0000-0003-0202-7816
  - name: Mark J. van der Laan
    affiliation: "5"
  - name: Ava Khamseh
    affiliation: "2, 3, 5"
    orcid: 0000-0001-5203-2205
  - name: Sjoerd Viktor Beentjes
    affiliation: "2, 4, 5"
    orcid: 0000-0002-7998-4262
affiliations:
  - name: Institute for Regeneration and Repair, University of Edinburgh, Edinburgh EH16 4UU, United Kingdom
    index: 1
  - name: MRC Human Genetics Unit, Institute of Genetics and Cancer, University of Edinburgh, Edinburgh EH4 2XU, United Kingdom.
    index: 2
  - name: School of Informatics, University of Edinburgh, Edinburgh EH8 9AB, United Kingdom
    index: 3
  - name: School of Mathematics and Maxwell Institute for Mathematical Sciences, University of Edinburgh, Edinburgh EH9 3FD, United Kingdom
    index: 4
  - name: Division of Biostatistics, University of California, Berkeley, CA, USA
    index: 5

date: 28 May 2025
bibliography: paper.bib
---

# Summary

TMLE.jl is a Julia package implementing targeted minimum loss-based estimation (TMLE), a general framework for causal effect estimation that unites modern machine learning with the theoretical guarantees of semiparametric statistics. TMLE yields doubly robust and semiparametrically efficient estimators, meaning it remains consistent if either the outcome model or the treatment assignment model is correctly specified, and it achieves the smallest possible asymptotic variance under standard regularity conditions. The package integrates with the broader Julia machine learning ecosystem and can be used in both observational and experimental settings. It is particularly well-suited for high-dimensional problems where robust inference is essential.

# Background

Causal inference is essential for understanding the effects of interventions in real-world settings, such as determining whether a treatment improves health outcomes or whether a genetic variant contributes to disease risk. Traditional approaches often begin by positing a specific parametric model, such as a linear-Gaussian or logistic regression model, and then estimating its parameters using efficient likelihood-based methods. This strategy has two main drawbacks. First, the quantity being estimated is often dictated by the parametric form of the model rather than the scientific question of interest (e.g., additive effects under a linear model, odds ratios under a logistic model). Second, when the model is misspecified, particularly in high-dimensional settings with complex interactions, parametric estimators can be severely biased.

Over the past two decades, new approaches have emerged that combine causal inference, machine learning (ML), and semiparametric theory to address these limitations. These methods begin with the explicit definition of a target parameter, often a causal estimand such as the average treatment effect, derived from a causal model. They then replace restrictive parametric models with flexible ML algorithms (e.g., gradient boosting, neural networks) to estimate nuisance components such as the outcome regression and treatment mechanism (called nuisance functions). However, since most ML methods are optimised for prediction rather than unbiased estimation, ML-based estimates are still biased when plugged directly into the parameter formula. To remove this bias, modern approaches use a debiasing step based on a key mathematical object, the efficient influence function (EIF). The result is an estimator that is asymptotically unbiased, efficient, and valid under much weaker assumptions than traditional parametric models.

Several estimators share this debiasing principle. The one-step estimator (OSE) [@pfanzagl1985contributions;@kennedy2024semiparametric] is a general methodology which proceeds by estimating and removing the bias resulting from machine-learning fitting via the EIF. The widely used augmented inverse probability of treatment weighting [@glynn2010introduction], is a special case of the OSE for the average treatment effect. The double machine learning (DML) framework [@chernozhukov2018double] extends the OSE by introducing cross-fitting, which allows the use of highly flexible ML algorithms without restrictive empirical process conditions (e.g., Donsker assumptions). A limitation of both OSE and DML is that they perform debiasing in the parameter space, which can lead to estimates outside the natural range of the target causal estimand (e.g., probabilities below 0 or above 1). In contrast, the targeted maximum likelihood estimator (TMLE) [@van2011targeted; @van2018targeted] performs the debiasing step in function space by iteratively updating the initial ML fit so that the resulting estimate both removes bias and respects the parameter’s natural bounds. While more involved to implement, TMLE retains all the theoretical guarantees of OSE/DML while ensuring parameter validity.

TMLE.jl is a Julia package that implements both TMLE and the OSE, with or without cross-fitting, for the estimation of causal parameters in observational and experimental studies. It enables researchers to estimate average treatment effects and other causal parameters while leveraging modern ML algorithms. TMLE.jl is applicable across a broad range of disciplines—including epidemiology, biostatistics, econometrics, and genomics—where valid causal effect estimation from high-dimensional or observational data is essential.

# Statement of Need

The main entry point to the DML methodology, in both Python and R, is the DoubleML package [@doubleml2024R;@DoubleML2022Python;@DoubleML]. Further Python packages focused on the estimation of the conditional average treatment effect include EconML [@econml] and CausalML [@chen2020causalml].

Despite its theoretical and practical advantages, targeted maximum likelihood estimation (TMLE) remains largely implemented in R, with limited support in other languages. The original tmle package [@tmleR] provides a single estimator for various causal parameters and relies on the SuperLearner package for flexible nuisance estimation. The more recent tmle3 package [@coyle2021tmle3-rpkg] offers a unified, object-oriented interface that explicitly represents the key mathematical components of TMLE. It supports a broad range of parameters, integrates cross-validated TMLE (CV-TMLE), the TMLE analogue of cross-fitting in DML, and is part of the broader tlverse ecosystem. Specialised extensions in this ecosystem target particular estimands, such as the mean outcome under the optimal treatment rule (tmle3mopttx) [@malenica2022tmle3mopttx] and stochastic intervention parameters (tmle3shift) [@hejazi2021tmle3shift-rpkg]. Longitudinal TMLE for time-varying exposures is supported by the separate ltmle package [@lendle2017ltmle]. To improve the robustness of estimators, for instance in the presence of practical violations or model instability, the collaborative targeted maximum likelihood estimation (C-TMLE) framework has been proposed, with several implementations available via the ctmle package [@ctmle].

For practitioners and developers who prefer a performant, composable, and type-safe environment such as Julia, no native TMLE implementation previously existed. As causal inference methods gain traction in computational biology, health sciences, economics, and other data-intensive disciplines, the absence of robust, well-integrated TMLE tooling in modern scientific programming languages has become increasingly limiting. TMLE.jl addresses this gap by providing the first native Julia implementation of TMLE. It supports the estimation of a variety of causal estimands, including the counterfactual mean, average treatment effect, and average interaction effects of arbitrary order. Any differentiable transformation of these estimands (e.g., risk ratio, odds ratio) can be obtained via automatic differentiation. Both TMLE and one-step estimators (OSE) are available in canonical and cross-fitting variants through a unified interface, and selected C-TMLE instantiations (greedy and scalable) are also implemented. The package accommodates more than binary treatments, allowing for any number of categorical treatments, an important feature for studying combinatorial intervention effects.

TMLE.jl is fully integrated into the broader Julia ecosystem. Machine learning models, including ensemble learners, can be specified via the MLJ toolbox [@blaom2020mlj]; datasets are represented as DataFrames [@bouchet2023dataframes]; and automatic differentiation is supported through any backend via DifferentiationInterface.jl [@dalle2025commoninterfaceautomaticdifferentiation;@schäfer2022abstractdifferentiationjlbackendagnosticdifferentiableprogramming]. Simulation datasets from CausalTables.jl [@Balkus2025] are also directly supported.

In doing so, TMLE.jl fills an important niche for causal inference in Julia, expanding the reach of TMLE beyond R and contributing to a growing ecosystem of open-source tools for rigorous, scalable, and reproducible statistical modelling.

# Applications to Population Genetics

While TMLE.jl is applicable to a broad range of scientific problems, it was developed with population genetics in mind, which informed several aspects of its design. Its performance has been benchmarked in large-scale population genetics simulations and applied to UK Biobank data [@labayle2025semi]. In such settings, the number of estimands can reach millions, posing a challenge for semiparametric estimators that rely on computationally intensive machine learning procedures. To address this, TMLE.jl implements automatic caching of intermediate results, enabling substantial computational savings. For example, when estimating the effect of a single treatment variable across multiple traits, the same propensity score can be reused across all analyses without recomputation. This mechanism has already been applied to the discovery of genetic variants affecting human traits via differential binding (submitted), and is currently being used in studies of genetic variants associated with myalgic encephalomyelitis.

# Acknowledgements

Olivier Labayle was supported by the United Kingdom Research and Innovation (grant EP/S02431X/1), UKRI Centre for Doctoral Training in Biomedical AI at the University of Edinburgh, School of Informatics.
Mark van der Laan is supported by NIH grant R01AI074345.
Chris P. Ponting was funded by the MRC (MC_UU_00007/15).
Ava Khamseh was supported by the XDF Programme from the University of Edinburgh and Medical Research Council (MC_UU_00009/2), and by a Langmuir Talent Development Fellowship from the Institute of Genetics and Cancer, and a philanthropic donation from Hugh and Josseline Langmuir.

# References {#references .unnumbered}
