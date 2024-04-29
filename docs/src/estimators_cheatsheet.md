# Estimators' Cheatsheet

This section is an effort to succinctly summarize the definition of semi-parametric estimators available in this package. As such, it is not self-contained, rather, it is intended as a mathematical memo that can be quickly searched. Gradients, One-Step and Targeted Maximum-Likelihood estimators are provided for the Counterfactual Mean, Average Treatment Effect and Average Interaction Effect. Estimators are presented in both their canonical and cross-validated versions.

One major difficulty I personally faced when entering the field, was the overwhelming notational burden. Unfortunately, this burden is necessary to understand how the various mathematical objects are handled by the procedures presented below. It is thus worth the effort to make sure you understand what each notation means. The reward? After reading this document, you should be able to implement any estimator present in this page.

Finally, if you find inconsistencies or imprecision, please report it, so we can keep improving!

## Where it all begins

### Notations

This is the notation we use throughout:

- The observed data: We assume we observe the realization of a random vector ``\bold{Z}_n = (Z_1, ..., Z_n)``. The components of ``\bold{Z}`` are assumed independent and identically distributed according to ``\mathbb{P}``, i.e. ``\forall i \in \{1, ..., n\},Z_i \sim \mathbb{P}``. Note that each ``Z_i`` is usually a vector as well, for us: ``Z_i = (W_i, T_i, Y_i)``.
- The Empirical Distribution : ``\mathbb{P}_n(A) = \frac{1}{n}\sum_{i=1}^n 1_A(Z_i)``. For each set ``A``, it computes the mean number of points falling into ``A``.
- Expected value of a function ``f`` of the data ``Z`` with respect to a distribution ``\mathbb{P}``: ``\mathbb{P}f \equiv \mathbb{E}_{\mathbb{P}}[f(Z)]``. Note that for the empirical distribution ``\mathbb{P}_n``, this is simply: ``\mathbb{P}_nf = \frac{1}{n}\sum_{i=1}^nf(Z_i)``, in other words, the sample mean.
- The Estimand: The unknown finite dimensional quantity we are interested in. ``\Psi`` is defined as a functional, i.e. ``\mathbb{P} \mapsto \Psi(\mathbb{P}) \in \mathbb{R}^d``. In this document ``d=1``.
- The Gradient of ``\Psi`` at the distribution ``\mathbb{P}`` : ``\phi_{\mathbb{P}}``, a function of the data, i.e. ``Z \mapsto \phi_{\mathbb{P}}(Z)``. The gradient satisfies two properties: (i) ``\mathbb{P}\phi_{\mathbb{P}} = 0``, and (ii) ``Var[\phi_{\mathbb{P}}] < \infty``.
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
CM_t(\mathbb{P}) = \Psi_t(\mathbb{P}) = \mathbb{E}_{\mathbb{P}}[\mathbb{E}_{\mathbb{P}}[Y | W, T = t]] = \int \mathbb{E}_{\mathbb{P}}[Y | W=w, T = t] d\mathbb{P}(w)
```

From this definition, we can see that ``\Psi_t`` depends on ``\mathbb{P}`` only through two relevant factors:

```math
\begin{aligned}
Q_Y(W, T) &= \mathbb{E}_{\mathbb{P}}[Y | W, T] \\
Q_W(W) &= \mathbb{P}(W)
\end{aligned}
```

So that ``\Psi(\mathbb{P}) = \Psi(Q_Y, Q_W)`` (the ``t`` subscript is dropped as it is unambiguous). This makes explicit that we don't need to estimate ``\mathbb{P}`` but only the relevant factors ``(Q_Y, Q_W)`` to obtain a plugin estimate of ``\Psi(\mathbb{P})``. Finally, it will be useful to define an additional factor:

```math
g(W, T) = \mathbb{P}(T | W) 
```

This is because the gradient of ``\Psi``:

```math
\phi_{CM_{t}, \mathbb{P}}(W, T, Y) = \frac{\mathbb{1}(T = t)}{g(W, t)}(Y − Q_Y(W, t)) + Q_Y(W, t) − \Psi(\mathbb{P}) 
```

which is the foundation of semi-parametric estimation, depends on this so-called nuisance parameter.

### Average Treatment Effect and Average Interaction Effect

They are simple linear combinations of counterfactual means and so are their gradients.

#### Average Treatment Effect

In all generality, for two values of a categorical treatment variable ``T: t_{control} \rightarrow t_{case}``, the Average Treatment Effect (ATE) is defined by:

```math
ATE_{t_{case}, t_{control}}(\mathbb{P}) = (CM_{t_{case}} - CM_{t_{control}})(\mathbb{P}) 
```

And it's associated gradient is:

```math
\begin{aligned}
\phi_{ATE}(W, T, Y) &= (\phi_{CM_{t_{case}}} - \phi_{CM_{t_{control}}})(W, T, Y) \\
&=  H(W, T)(Y − Q_Y(W, T)) + C(W) − ATE_{t_{case}, t_{control}}(\mathbb{P})
\end{aligned}
```

with:

```math
\begin{aligned}
  \begin{cases}
    H(W, T) &= \frac{ \mathbb{1}(T = t_{case}) - \mathbb{1}(T = t_{control})}{g(W, T)} \\
    C(W) &= (Q_Y(W, t_{case}) - Q_Y(W, t_{control}))
  \end{cases}
\end{aligned}
```

``H`` is also known as the clever covariate in the Targeted Learning literature (see later).

#### Average Interaction Effect

For simplicity, we only consider two treatments ``T = (T_1, T_2)`` such that ``T_1: t_{1,control} \rightarrow t_{1,case}`` and ``T_2: t_{2,control} \rightarrow t_{2,case}``. The Average Interaction Effect (AIE) of ``(T_1, T_2)`` is defined as:

```math
\begin{aligned}
AIE(\mathbb{P}) &= (CM_{t_{1, case}, t_{2, case}} - CM_{t_{1, case}, t_{2, control}} \\
&- CM_{t_{1, control}, t_{2, case}} + CM_{t_{1, control}, t_{2, control}})(\mathbb{P})
\end{aligned}
```

And its gradient is given by:

```math
\begin{aligned}
\phi_{AIE, }(W, T, Y) &= (\phi_{CM_{t_{1,case}, t_{2,case}}} - \phi_{CM_{t_{1,case}, t_{2,control}}} \\
&- \phi_{CM_{t_{1,control}, t_{2,case}}} + \phi_{CM_{t_{1,control}, t_{2,control}}})(W, T, Y) \\
&=  H(W, T)(Y − Q_Y(W, T)) + C(W) − ATE_{t_{case}, t_{control}}(\mathbb{P})
\end{aligned}
```

with:

```math
\begin{aligned}
  \begin{cases}
    H(W, T) &= \frac{(\mathbb{1}(T_1 = t_{1, case}) - \mathbb{1}(T_1 = t_{1, control}))(\mathbb{1}(T_2 = t_{2, case}) - \mathbb{1}(T_2 = t_{2, control}))}{g(W, T)} \\
    C(W) &= (Q_Y(W, t_{1, case}, t_{2, case}) - Q_Y(W, t_{1, case}, t_{2, control}) - Q_Y(W, t_{1, control}, t_{2, case}) + Q_Y(W, t_{1, control}, t_{2, control}))
  \end{cases}
\end{aligned}
```

For higher-order interactions the two factors ``H`` and ``C`` can be similarly inferred.

## Asymptotic Analysis of Plugin Estimators

The theory is based on the von Mises expansion (functional equivalent of Taylor's expansion) which reduces due to the fact that the gradient satisfies: ``\mathbb{E}_{\mathbb{P}}[\phi_{\mathbb{P}}(Z)] = 0``.

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

### 1. The Asymptotically Linear Term

```math
\mathbb{P}_n\phi_{\mathbb{P}}(Z)
```

By the central limit theorem, it is asymptotically normal with variance ``Var[\phi]/n``, it will be used to construct a confidence interval for our final estimate.

### 2. The First-Order Bias Term

```math
- \mathbb{P}_n\phi_{\mathbb{\hat{P}}}(Z)
```

This is the term both the One-Step estimator and the Targeted Maximum-Likelihood estimator deal with.

### 3. The Empirical Process Term

```math
(\mathbb{P}_n − \mathbb{P})(ϕ_{\mathbb{\hat{P}}}(Z) − \phi_{\mathbb{P}}(Z))
```

This can be shown to be of order ``o_{\mathbb{P}}(\frac{1}{\sqrt{n}})`` if ``\phi_{\hat{\mathbb{P}}}`` converges to ``\phi_{\mathbb{P}}`` in ``L_2(\mathbb{P})`` norm, that is:

```math
\int (\phi_{\hat{\mathbb{P}}}(Z) - \phi_{\mathbb{P}}(Z) )^2 d\mathbb{P}(Z) = o_{\mathbb{P}}\left(\frac{1}{\sqrt{n}}\right)
```

and, any of the following holds:

- ``\phi`` (or equivalently its components) is [[Donsker](https://en.wikipedia.org/wiki/Donsker_classes)], i.e., not too complex.
- The estimator is constructed using sample-splitting (see cross-validated estimators).

### 4. The Exact Second-Order Remainder Term

```math
R_2(\mathbb{\hat{P}}, \mathbb{P})
```

This term is usually more complex to analyse. Note however, that it is entirely defined by the von Mises expansion, and for the counterfactual mean, it can be shown that if ``g(W,T) \geq \frac{1}{\eta}`` (positivity constraint):

```math
|R_2(\mathbb{\hat{P}}, \mathbb{P})| \leq \eta ||\hat{Q}_{n, Y} - Q_{Y}|| \cdot ||\hat{g}_{n} - g||
```

and thus, if the estimators ``\hat{Q}_{n, Y}`` and ``\hat{g}_{n}`` converge at a rate ``o_{\mathbb{P}}(n^{-\frac{1}{4}})``, the second-order remainder will be ``o_{\mathbb{P}}(\frac{1}{\sqrt{n}})``. This is the case for many popular estimators like random forests, neural networks, etc...

## Asymptotic Linearity and Inference

According to the previous section, the OSE and TMLE will be asymptotically linear with efficient influence curve the gradient ``\phi``:

```math
\sqrt{n}(\hat{\Psi} - \Psi) = \frac{1}{\sqrt{n}} \sum_{i=1}^n \phi(Z_i) + o_{\mathbb{P}}\left(\frac{1}{\sqrt{n}}\right)
```

By the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) and [Slutsky's Theorem](https://en.wikipedia.org/wiki/Slutsky%27s_theorem) we have:

```math
\sqrt{n}(\hat{\Psi} - \Psi) \leadsto \mathcal{N}(0, Std(\phi))
```

Now, if we consider ``S_n = \hat{Std}(\phi_{\hat{\mathbb{P}}})`` as a consistent estimator for ``Std(\phi)``, we have by Slutsky's Theorem again, the following pivot:

```math
\frac{\sqrt{n}(\hat{\Psi} - \Psi)}{S_n} \leadsto \mathcal{N}(0, 1)
```

which can be used to build confidence intervals:

```math
\underset{n \to \infty}{\lim} P(\hat{\Psi}_n - \frac{S_n}{\sqrt{n}}z_{\alpha} \leq \Psi \leq \hat{\Psi}_n + \frac{S_n}{\sqrt{n}}z_{\alpha}) = 1 - 2\alpha
```

Here, ``z_{\alpha}`` denotes the ``\alpha``-quantile function of the standard normal distribution

## One-Step Estimator

### Canonical OSE

The One-Step estimator is very intuitive, it simply corrects the initial plugin estimator by adding in the residual bias term. As such, it corrects for the bias in the estimand's space. Let ``\hat{P}= (\hat{Q}_{n,Y}, \hat{Q}_{n,W})`` be an estimator of the relevant factors of ``\mathbb{P}`` as well as ``\hat{g}_n`` an estimator of the nuisance function ``g``. The OSE is:

```math
\hat{\Psi}_{n, OSE} = \Psi(\hat{P}) + \mathbb{P}_n{\phi_{\hat{P}}}
```

### CV-OSE

Instead of assuming Donsker conditions limiting the complexity of the algorithms used, we can use sample splitting techniques. For this, we split the data into K folds of (roughly) equal size. For a given sample i, we denote by ``k(i)`` the fold it belongs to (called validation fold/set) and by ``-k(i)`` the union of all remaining folds (called training set). Similarly, we denote by ``\hat{Q}^{k}`` an estimator for ``Q`` obtained from samples in the validation fold ``k`` and ``\hat{Q}^{-k}`` an estimator for ``Q`` obtained from samples in the (training) fold ``\{1, ..., K\}-\{k\}``.

The cross-validated One-Step estimator can be compactly written as an average over the folds of sub one-step estimators:

```math
\begin{aligned}
\hat{\Psi}_{n, CV-OSE} &= \sum_{k=1}^K \frac{N_k}{n} (\Psi(\hat{Q}_Y^{-k}, \hat{Q}_W^{k}) + \hat{\mathbb{P}}_n^k \phi^{-k}) \\
&= \frac{1}{n} \sum_{k=1}^K \sum_{\{i: k(i) = k\}} (\hat{Q}_Y^{-k}(W_i, T_i) + \phi^{-k}(W_i, T_i, Y_i))
\end{aligned}
```

Where the first equation is the general form of the estimator while the second one corresponds to the counterfactual mean. The important thing to note is that for each sub one-step estimator, the sum runs over the validation samples while ``\hat{Q}_Y^{-k}`` and ``\hat{\phi}^{-k}`` are estimated using the training samples.

## Targeted Maximum-Likelihood Estimator

Unlike the One-Step estimator, the Targeted Maximum-Likelihood Estimator corrects the bias term in distribution space. That is, it moves the initial estimate ``\hat{\mathbb{P}}^0=\hat{\mathbb{P}}`` to a corrected ``\hat{\mathbb{P}}^*`` (notice the new superscript notation). Then the plugin principle can be applied and the targeted estimator is simply ``\hat{\Psi}_{n, TMLE} = \Psi(\hat{\mathbb{P}}^*)``. This means TMLE always respects the natural range of the estimand, giving it an upper hand on the One-Step estimator.

### Canonical TMLE

The way ``\hat{\mathbb{P}}`` is modified is by means of a parametric sub-model also known as a fluctuation. The choice of fluctuation depends on the target parameter of interest. It can be shown that for the conditional mean, it is sufficient to fluctuate ``\hat{Q}_{n, Y}`` only once using the following fluctuations:

- ``\hat{Q}_{Y, \epsilon}(W, T) = \hat{Q}_{n, Y}(T, W) + \epsilon \hat{H}(T, W)``, for continuous outcomes ``Y``.
- ``\hat{Q}_{Y, \epsilon}(W, T) = \frac{1}{1 + e^{-(logit(\hat{Q}_{n, Y}(T, W)) + \epsilon \hat{H}(T, W))}}``, for binary outcomes ``Y``.

where ``\hat{H}(T, W) = \frac{1(T=t)}{\hat{g}_n(W)}`` is known as the clever covariate. The value of ``\epsilon`` is obtained by minimizing the loss ``L`` associated with ``Q_Y``, that is the mean-squared error for continuous outcomes and negative log-likelihood for binary outcomes. This can easily be done via linear and logistic regression respectively, using the initial fit as off-set.

!!! note
    For the ATE and AIE, just like the gradient is linear in ``\Psi``, the clever covariate used to fluctuate the initial ``\hat{Q}_{n, Y}`` is as presented in [Average Treatment Effect and Average Interaction Effect](@ref)

If we denote by ``\epsilon^*`` the value of ``\epsilon`` minimizing the loss, the TMLE is:

```math
\hat{\Psi}_{n, TMLE} = \Psi(\hat{Q}_{n, Y, \epsilon^*}, \hat{Q}_{n, W})
```

### CV-TMLE

Using the same notation as the cross-validated One-Step estimator, the fluctuated distribution is obtained by solving:

```math
\epsilon^* = \underset{\epsilon}{\arg \min} \frac{1}{n} \sum_{k=1}^K \sum_{\{i: k(i) = k\}} L(Y_i, \hat{Q}_{Y, \epsilon}^{-k}(W_i, T_i, Y_i))
```

where ``\hat{Q}_{Y, \epsilon}`` and ``L`` are the respective fluctuations and loss for continuous and binary outcomes. This leads to a targeted ``\hat{Q}_{n,Y}^{*}`` such that:

```math
\forall i \in \{1, ..., n\}, \hat{Q}_{n, Y}^{*}(W_i, T_i) = \hat{Q}_{Y, \epsilon^*}^{-k(i)}(W_i, T_i)
```

That is, the predictions of ``\hat{Q}_{n, Y}^{*}`` for sample ``i`` are based on the out of fold predictions of ``\hat{Q}_{Y}^{-k(i)}`` and the "pooled" fluctuation given by ``\epsilon^*``.

Then, the CV-TMLE is:

```math
\begin{aligned}
\hat{\Psi}_{n, CV-TMLE} &= \sum_{k=1}^K \frac{N_k}{n} \Psi(\hat{Q}_{n, Y}^{*}, \hat{Q}_W^{k}) \\
&= \frac{1}{n} \sum_{k=1}^K \sum_{\{i: k(i) = k\}} \hat{Q}_{n, Y}^{*}(W_i, T_i)
\end{aligned}
```

Notice that, while ``\hat{\Psi}_{n, CV-TMLE}`` is not a plugin estimator anymore, it still respects the natural range of the parameter because it is an average of plugin estimators.

## References

The content of this page is largely inspired from:

- [Semiparametric doubly robust targeted double machine learning: a review](https://arxiv.org/pdf/2203.06469.pdf).
- [Introduction to Modern Causal Inference](https://alejandroschuler.github.io/mci/).
- [Targeted Learning, Causal Inference for Observational and Experimental Data](https://link.springer.com/book/10.1007/978-1-4419-9782-1).
- [STATS 361: Causal Inference](https://web.stanford.edu/~swager/stats361.pdf)
