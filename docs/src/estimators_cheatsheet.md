# Estimators' Cheatsheet

This section is an effort to succinctly summarize the definition of semi-parametric estimators available in this package. As such, it is not self-contained, rather, it is intended as a mathematical memo that can be quickly searched. Gradients, One-Step and Targeted Maximum-Likelihood estimators are provided for the Counterfactual Mean, Average Treatment Effect and Average Interaction Effect. Estimators are presented in both their canonical and cross-validated versions.

One major difficulty I personally faced when entering the field, was the overwhelming notational burden. Unfortunately, this burden is necessary to understand how the various mathematical objects are handled by the procedures presented below. It is thus worth the effort to make sure you understand what each notation means. The reward? After reading this document, you should be able to implement any estimator present in this page.

Finally, if you find inconsistencies or imprecision, please report it, so we can keep improving!

!!! note
    This page is still under construction and more content will be added in the coming months.

## Where it all begins

### Notations

This is the notation we use throughout:

- The observed data: We assume we observe the realization of a random vector ``\bold{Z}_n = (Z_1, ..., Z_n)``. The components of ``\bold{Z}`` are assumed independent and identically distributed according to ``\mathbb{P}``, i.e. ``\forall i \in \{1, ..., n\},Z_i \sim \mathbb{P}``. Note that each ``Z_i`` is usually a vector as well, for us: ``(W_i, T_i, Y_i)``.
- The Empirical Distribution : ``\mathbb{P}_n(A) = \frac{1}{n}\sum_{i=1}^n 1_A(Z_i)``. For each set ``A``, it computes the mean number of points falling into ``A``.
- Expected value of a function ``f`` with respect to a distribution ``\mathbb{P}``: ``\mathbb{E}_{\mathbb{P}} f = \mathbb{P}f``. Note that for the empirical distribution ``\mathbb{P}_n``, this is simply: ``\mathbb{P}_nf = \frac{1}{n}\sum_{i=1}^nf(Z_i)``, in other words, the sample mean.
- The Estimand: The unknown finite dimensional quantity we are interested in. ``\Psi`` is defined as a functional, i.e. ``\mathbb{P} \mapsto \Psi(\mathbb{P}) \in \mathbb{R}^d``. In this document ``d=1``.
- The Gradient of ``\Psi`` at the distribution ``\mathbb{P}`` : ``\phi_{\mathbb{P}}``, a function of the data, i.e. ``Z \mapsto \phi_{\mathbb{P}}(Z)``. The gradient satisfies: ``\mathbb{P}\phi_{\mathbb{P}} = 0`` and ``Var[\phi_{\mathbb{P}}] < \infty``.
- An estimator, is a function of the data ``\bold{Z}_n``, and thus a random variable, that seeks to approximate an unknown quantity. For instance, ``\hat{\Psi}_n`` denotes an estimator for ``\Psi``. Notice that the empirical distribution is an estimator for ``\mathbb{P}``, but the hat is omitted to distinguish it from other estimators that will be defined later on.

### The Counterfactual Mean

Let ``\bold{Z}=(W, T, Y)_{i=1..n}`` be a dataset generated according to the following structural causal model:

```math
\begin{aligned}
W &= f_W(U_W) \\
T &= f_T(W, U_T) \\
Y &= f_Y(T, W, U_Y)
\end{aligned}
```

This is certainly the most common scenario in causal inference where ``Y`` is an outcome of interest, ``T`` a set of treatment variables and ``W`` a set of confounding variables. We are generally interested in the effect of ``T`` on ``Y``. In this document we will consider the counterfactual mean as a workhorse. We will show that the estimators for the Average Treatment Effect and Average Interaction Effect can easily be derived from it. Under usual conditions, the counterfactual mean identifies to the following statistical estimand:

```math
CM_t(\mathbb{P}) = \Psi_t(\mathbb{P}) = \mathbb{E}_{\mathbb{P}}[\mathbb{E}_{\mathbb{P}}[Y | W, T = t]] = \int \mathbb{E}_{\mathbb{P}}[Y | W, T = t] d\mathbb{P}(w)
```

From this definition, we can see that ``\Psi_t`` depends on ``\mathbb{P}`` only through two relevant factors:

```math
\begin{aligned}
Q_Y(W, T) &= \mathbb{E}_{\mathbb{P}}[Y | W, T] \\
Q_W(W) &= \mathbb{P}(W)
\end{aligned}
```

So that ``\Psi(\mathbb{P}) = \Psi(Q_Y, Q_W)``, which makes explicit that we don't need to estimate ``\mathbb{P}`` but only the relevant factors ``(Q_Y, Q_W)`` to obtain a plugin estimate of ``\Psi(\mathbb{P})``. Finally, it will be useful to define an additional factor:

```math
g(W, T) = \mathbb{P}(T | W) 
```

Since the gradient of ``\Psi``:

```math
\phi_{CM_{t}, \mathbb{P}}(W, T, Y) = \frac{\mathbb{1}(T = t)}{g(W, t)}(Y − Q_Y(W, t)) + Q_Y(W, t) − \Psi(\mathbb{P}) 
```

which is the foundation of semi-parametric estimation, depends on this so-called nuisance parameter.

### Average Treatment Effect and Average Interaction Effect?

They are simple linear combinations of counterfactual means and so is their gradients.

#### Average Treatment Effect

For two values of ``T: t_{control} \rightarrow t_{case}``, the Average Treatment Effect (ATE) is defined by:

```math
ATE_{t_{case}, t_{control}}(\mathbb{P}) = CM_{t_{case}}(\mathbb{P}) - CM_{t_{control}}(\mathbb{P}) 
```

And it's associated gradient is:

```math
\begin{aligned}
\phi_{ATE}(W, T, Y) &= \phi_{CM_{t_{case}}}(W, T, Y) - \phi_{CM_{t_{control}}}(W, T, Y) \\
&=  H(W, T)(Y − Q_Y(W, T)) + C(W) − ATE_{t_{case}, t_{control}}(\mathbb{P})
\end{aligned}
```

with:

```math
\begin{equation}
  \begin{cases}
    H(W, T) &= \frac{(-\mathbb{1}(T \in (t_{case}, t_{control})))^{T=t_{control}}}{g(W, T)} \\
    C(W) &= (Q_Y(W, t_{case}) - Q_Y(W, t_{control}))
  \end{cases}
\end{equation}
```

``H`` is also known as the clever covariate in the Targeted Learning literature (see later).

#### Average Interaction Effect

For simplicity we only consider two treatments ``T_1: t_{1,control} \rightarrow t_{1,case}`` and ``T_2: t_{2,control} \rightarrow t_{2,case}``.

```math
AIE(\mathbb{P}) = CM_{t_{1, case}, t_{2, case}}(\mathbb{P}) - CM_{t_{1, case}, t_{2, control}}(\mathbb{P}) - CM_{t_{1, control}, t_{2, case}}(\mathbb{P}) + CM_{t_{1, control}, t_{2, control}}(\mathbb{P})
```

The gradient is similarly given by:

```math
\begin{aligned}
\phi_{AIE, }(W, T, Y) &= \phi_{CM_{t_{1,case}, t_{2,case}}}(W, T, Y) - \phi_{CM_{t_{1,case}, t_{2,control}}}(W, T, Y) - \phi_{CM_{t_{1,control}, t_{2,case}}}(W, T, Y) + \phi_{CM_{t_{1,control}, t_{2,control}}}(W, T, Y) \\
&=  H(W, T)(Y − Q_Y(W, T)) + C(W) − ATE_{t_{case}, t_{control}}(\mathbb{P})
\end{aligned}
```

with:

```math
\begin{equation}
  \begin{cases}
    H(W, T) &= \frac{(-\mathbb{1}(T \in (t_{case}, t_{control})))^{T=t_{control}}}{g(W, T)} \\
    C(W) &= (Q_Y(W, t_{case}) - Q_Y(W, t_{control}))
  \end{cases}
\end{equation}
```

## Motivation for the Estimators

The theory is based on the following von Mises expansion (functional equivalent of Taylor's expansion) which reduces due to the fact that the gradient satisfies: ``\mathbb{E}_{\mathbb{P}}[\phi_{\mathbb{P}}(Z)] = 0``.

```math
\begin{aligned}
\Psi(\hat{\mathbb{P}}) − \Psi(\mathbb{P}) &= \int \phi_{\hat{\mathbb{P}}}(z) \mathrm{d}(\hat{\mathbb{P}} − \mathbb{P})(z) + R_2(\hat{\mathbb{P}}, \mathbb{P}) \\
&= - \int \phi_{\hat{\mathbb{P}}}(z) \mathrm{d}\mathbb{P}(z) + R_2(\hat{\mathbb{P}}, \mathbb{P})
\end{aligned}
```

This suggests that a plugin estimator, one that simply evaluates ``\Psi`` at an estimator ``\hat{\mathbb{P}}`` of ``\mathbb{P}``, is biased. This bias can be elegantly decomposed in four terms by reworking the previous expression:

```math
\Psi(\hat{\mathbb{P}}) − \Psi(\mathbb{P}) = \mathbb{P}_n\phi_{\mathbb{P}}(Z) - \mathbb{P}_n\phi_{\mathbb{\hat{P}}}(Z) + (\mathbb{P}_n − \mathbb{P})(ϕ_{\mathbb{\hat{P}}}(Z) − \phi_{\mathbb{P}}(Z)) + R_2(\mathbb{\hat{P}}, \mathbb{P})
```

- The asymptotically linear term:

```math
\mathbb{P}_n\phi_{\mathbb{P}}(Z)
```

By the central limit theorem, it is asymptotically normal with variance ``Var[\phi]/n``, it is used to build confidence interval.

- The bias term:

```math
- \mathbb{P}_n\phi_{\mathbb{\hat{P}}}(Z)
```

This is the term both the One-Step estimator and the Targeted Maximum-Likelihood estimator deal with.

- The empirical process term:

```math
(\mathbb{P}_n − \mathbb{P})(ϕ_{\mathbb{\hat{P}}}(Z) − \phi_{\mathbb{P}}(Z))
```

This is usually negligible under minimal assumptions.

- The second-order remainder term:

```math
R_2(\mathbb{\hat{P}}, \mathbb{P})
```

This is often negligible, but see the cross-validated estimators section.

## Inference

According to the previous section, the OSE and TMLE will be asymptotically linear with efficient influence curve the gradient ``\phi``. This means the central limit theorem applies and the variance of the estimators can be estimated by:

```math
\begin{aligned}
\hat{Var}(\hat{\Psi}_n) &= \frac{\hat{Var}(\hat{\phi}_n)}{n} \\
&= \frac{1}{n(n-1)}\sum_{i=1}^n \hat{\phi}(W_i, T_i, Y_i)^2
\end{aligned}
```

because the gradient has mean 0.

## One-Step Estimator

### Canonical OSE

The One-Step estimator is very intuitive, it simply corrects the initial plugin estimator by adding in the residual bias term. As such, it corrects for the bias in the estimand's space:

```math
\hat{\Psi}_{n, OSE} = \Psi(\hat{P}) + P_n{\phi_{\hat{P}}(Z)}
```

### CV-OSE

Assume the realization of ``n`` random variables ``K_i`` assigning each sample to one of ``{1, ..., K}`` folds. For a given sample ``i``, we denote by ``k(i)`` the validation fold it belongs to and ``-k(i)`` the remaining training fold. Similarly, we denote by ``\hat{Q}^{k}`` an estimator for ``Q`` obtained from samples in the validation fold ``k`` and ``\hat{Q}^{-k}`` an estimator for ``Q`` obtained from samples in the (training) fold ``\{1, ..., K\}-\{k\}``.

The cross-validated One-Step estimator can be compactly written as an average over the folds of sub one-step estimators:

```math
\begin{aligned}
\hat{\Psi}_{n, CV-OSE} &= \sum_{k=1}^K \frac{N_k}{n} (\Psi(\hat{Q}_Y^{-k}, \hat{Q}_W^{k}) + \hat{\mathbb{P}}_n^k \phi^{-k}) \\
&= \frac{1}{n} \sum_{k=1}^K \sum_{\{i: k(i) = k\}} (\hat{Q}_Y^{-k}(W_i, T_i) + \phi^{-k}(W_i, T_i, Y_i))
\end{aligned}
```

The important thing to note is that for each sub one-step estimator, the sum runs over the validation samples while ``\hat{Q}_Y^{-k}`` and ``\hat{\phi}^{-k}`` are estimated using the training samples.

## Targeted Maximum-Likelihood Estimation

Unlike the One-Step estimator, the Targeted Maximum-Likelihood Estimator corrects the bias term in distribution space. That is, it moves the initial estimate ``\hat{\mathbb{P}}^0=\hat{\mathbb{P}}`` to a corrected ``\hat{\mathbb{P}}^*`` (notice the new superscript notation). Then the plugin principle can be applied and the targeted estimator is simply ``\hat{\Psi}_{n, TMLE} = \Psi(\hat{\mathbb{P}}^*)``. This means TMLE always respects the natural range of the estimand, giving it an upper hand on the One-Step estimator.

### Canonical TMLE

The way ``\hat{\mathbb{P}}`` is modified is by means of a parametric sub-model also known as a fluctuation. It can be shown that for the conditional mean, it is sufficient to fluctuate ``\hat{Q}_{n, Y}``. The fluctuations that are used in practice are:

- ``\hat{Q}_{Y, \epsilon}(W, T) = \hat{Q}_{n, Y}(T, W) + \epsilon \hat{H}(T, W)``, for continuous outcomes ``Y``.
- ``\hat{Q}_{Y, \epsilon}(W, T) = expit(logit(\hat{Q}_{n, Y}(T, W)) + \epsilon \hat{H}(T, W))``, for binary outcomes ``Y``.

where ``\hat{H}(T, W) = \frac{1(T=t)}{\hat{g}_n(W)}`` is known as the clever covariate. The value of ``\epsilon`` is obtained by minimizing the loss ``L`` associated with ``Q_Y``, that is the mean-squared error for continuous outcomes and negative log-likelihood for binary outcomes. This can easily be done via linear and logistic regression respectively.

!!! note
    Just like the gradient is linear in ``\Psi``, the clever covariate used to fluctuate the initial ``\hat{Q}_Y`` is as presented in [Average Treatment Effect and Average Interaction Effect?](@ref)

### CV-TMLE

Using the same notation as the cross-validated One-Step estimator, the fluctuated distribution is obtained by solving:

```math
\epsilon^* = \underset{\epsilon}{\arg \min} \frac{1}{n} \sum_{k=1}^K \sum_{\{i: k(i) = k\}} L(Y_i, \hat{Q}_{Y, \epsilon}^{-k}(W_i, T_i, Y_i))
```

where ``\hat{Q}_{Y, \epsilon}`` and ``L`` are the respective fluctuations and loss for continuous and binary outcomes. This leads to a targeted ``\hat{Q}_Y^{*}`` such that:

```math
\forall i, \hat{Q}_Y^{*}(W_i, T_i) = \hat{Q}_{Y, \epsilon^*}^{-k(i)}(W_i, T_i)
```

That is, the predictions of ``\hat{Q}_Y^{*}`` for sample ``i`` are based on the

Then, the CV-TMLE is:

```math
\begin{aligned}
\hat{\Psi}_{n, CV-TMLE} &= \sum_{k=1}^K \frac{N_k}{n} \Psi(\hat{Q}_Y^{*}, \hat{Q}_W^{k}) \\
&= \frac{1}{n} \sum_{k=1}^K \sum_{\{i: k(i) = k\}} \hat{Q}_Y^{*}(W_i)
\end{aligned}
```

Notice that while ``\hat{\Psi}_{n, CV-TMLE}`` is not a plugin estimator anymore, it still respects the natural range of the parameter because it is an average of plugin estimators. Also, because ``\hat{Q}_Y^{*,-k}`` is based both on training and validation samples, the elements of the sum are not truly independent.

## Acknowledgements

The content of this page was largely inspired from:

- [Semiparametric doubly robust targeted double machine learning: a review](https://arxiv.org/pdf/2203.06469.pdf).
- [Introduction to Modern Causal Inference](https://alejandroschuler.github.io/mci/).
- [Targeted Learning, Causal Inference for Observational and Experimental Data](https://link.springer.com/book/10.1007/978-1-4419-9782-1).
