---
title: 'TMLE.jl: Targeted Minimum Loss-Based Estimation In Julia.'
tags:
  - julia
  - statistics
  - semi-parametric statistics
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
  - name: School of Mathematics and Maxwell Institute, University of Edinburgh, Edinburgh EH9 3FD, United Kingdom
    index: 4
  - name: Division of Biostatistics, University of California, Berkeley, CA, USA
    index: 5

date: 28 May 2025
bibliography: paper.bib
---

# Summary

Causal inference is essential for understanding the effect of interventions in real-world settings, such as determining whether a treatment improves health outcomes or whether a gene variant contributes to disease risk. Traditional statistical methods, such as linear regression or propensity score matching, often rely on strong modeling assumptions and may fail to provide valid inference when these assumptions are violated—particularly in the presence of high-dimensional data or model misspecification.

TMLE.jl is a Julia package that implements Targeted Maximum Likelihood Estimation (TMLE) [@van2011targeted;@van2018targeted], a general framework for causal effect estimation that combines machine learning with principles from semiparametric statistics. TMLE provides doubly robust, efficient, and flexible estimation of causal parameters in observational and experimental studies.

The goal of TMLE.jl is to provide an accessible, implementation of this methodology within the Julia ecosystem. It enables researchers to estimate average treatment effects and other causal parameters while leveraging modern machine learning algorithms to flexibly model nuisance components, such as the outcome regression and treatment mechanism.

TMLE.jl is useful in a wide range of scientific disciplines—including epidemiology, biostatistics, econometrics, and genomics—where estimating causal effects from high-dimensional or observational data is critical [@smith2023application;@labayle2025semi;@gruber2010application]. Unlike traditional regression approaches, TMLE can incorporate nonparametric learning without sacrificing statistical validity, offering both robustness to model misspecification and valid confidence intervals.

# Statement of Need

Despite its theoretical and practical advantages, TMLE has limited implementation outside of R (notably the [tmle](https://cran.r-project.org/web/packages/tmle/index.html), [`tmle3`](https://github.com/tlverse/tmle3/blob/master/README.Rmd) and [ltmle](https://cran.r-project.org/web/packages/ltmle/index.html) packages [@tmleR;@coyle2021tmle3-rpkg;@lendle2017ltmle]). This limits accessibility for users and developers who prefer or require a performant, composable, and type-safe environment like Julia. As causal inference tools are increasingly used in computational biology, health sciences, economics, and beyond, there is a growing need for robust, well-integrated TMLE implementations in modern scientific programming languages.

TMLE.jl addresses this gap by providing the first native Julia implementation of TMLE. The main features of the package are:

* Estimation of classic counterfactual estimands: 
  * Counterfactual Mean (`CM`)
  * Average Treatment Effect (`ATE`)
  * First implementation of the Average Interaction Effect (`AIE`) up to any order.
  * Any differentiable function thereof via automatic differentiation. 
* Various semi-parametric estimators: 
  * Targeted Maximum Likelihood Estimators (canonical, weighted, cross-validated and two collaborative flavours)
  * One-Step Estimators (canonical and cross-validated).
* Support for combinations of factorial treatment variables.
* Integration with Julia’s ecosystem: 
  * Machine Learning models including ensemble learning via the [MLJ](https://juliaai.github.io/MLJ.jl/stable/) toolbox [@blaom2020mlj].
  * Dataset Representation with [DataFrames.jl](https://dataframes.juliadata.org/stable/) [@bouchet2023dataframes].
  * Automatic Differentiation via [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl)[@dalle2025commoninterfaceautomaticdifferentiation;@schäfer2022abstractdifferentiationjlbackendagnosticdifferentiableprogramming]

TMLE.jl fills an important niche for causal inference practitioners in Julia and contributes to the growing ecosystem of open-source tools supporting rigorous and scalable statistical modeling.

# Mentions

TMLE.jl is already being used in three large scale genomic projects:

- The evaluation of semi-parametric methods in population genetics with application to UK-Biobank data [@labayle2025semi]
- The discovery of genetic variants affecting human traits via differential binding (ongoing)
- The discovery of genetic variants associated with Myalgic encephalomyelitis (ongoing)

# Acknowledgements

Olivier Labayle was supported by the United Kingdom Research and Innovation (grant EP/S02431X/1), UKRI Centre for Doctoral Training in Biomedical AI at the University of Edinburgh, School of Informatics.
Mark van der Laan is supported by NIH grant R01AI074345.
Chris P. Ponting was funded by the MRC (MC_UU_00007/15).
Ava Khamseh was supported by the XDF Programme from the University of Edinburgh and Medical Research Council (MC_UU_00009/2), and by a Langmuir Talent Development Fellowship from the Institute of Genetics and Cancer, and a philanthropic donation from Hugh and Josseline Langmuir.

# References {#references .unnumbered}
