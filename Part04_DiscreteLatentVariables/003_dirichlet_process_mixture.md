---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"colab_type": "text", "id": "6oXxxX9LZL-h"}

# Dirichlet Process Mixture Models in Pyro


## What are Bayesian nonparametric models?
Bayesian nonparametric models are models where the number of parameters grow freely with the amount of data provided; thus, instead of training several models that vary in complexity and comparing them, one is able to design a model whose complexity grows as more data are observed. The prototypical example of Bayesian nonparametrics in practice is the *Dirichlet Process Mixture Model* (DPMM). A DPMM allows for a practitioner to build a mixture model when the number of distinct clusters in the geometric structure of their data is unknown – in other words, the number of clusters is allowed to grow as more data is observed. This feature makes the DPMM highly useful towards exploratory data analysis, where few facets of the data in question are known; this presentation aims to demonstrate this fact.

## The Dirichlet Process (Ferguson, 1973)
Dirichlet processes are a family of probability distributions over discrete probability distributions. Formally, the Dirichlet process (DP) is specified by some base probability distribution $G_0: \Omega \to \mathbb{R}$ and a positive, real, scaling parameter commonly denoted as $\alpha$. A sample $G$ from a Dirichlet process with parameters $G_0: \Omega \to \mathbb{R}$ and $\alpha$ is itself a distribution over $\Omega$. For any disjoint partition $\Omega_1, ..., \Omega_k$ of $\Omega$, and any sample $G \sim DP(G_0, \alpha)$, we have:

$$(G(\Omega_1), ..., G(\Omega_k)) \sim \text{Dir}(\alpha G_0(\Omega_1), ..., \alpha G_0(\Omega_k))$$

Essentially, this is taking a discrete partition of our sample space $\Omega$ and subsequently constructing a discrete distribution over it using the base distribution $G_0$. While quite abstract in formulation, the Dirichlet process is very useful as a prior in various graphical models. This fact becomes easier to see in the following scheme.

## The Chinese Restaurant Process (Aldous, 1985)

Imagine a restaurant with infinite tables (indexed by the positive integers) that accepts customers one at a time. The $n$th customer chooses their seat according to the following probabilities:

*   With probability $\frac{n_t}{\alpha + n - 1}$, sit at table $t$, where $n_t$ is the number of people at table $t$
*   With probability $\frac{\alpha}{\alpha + n - 1}$, sit at an empty table

If we associate to each table $t$ a draw from a base distribution $G_0$ over $\Omega$, and then associate unnormalized probability mass $n_t$ to that draw, the resulting distribution over $\Omega$ is equivalent to a draw from a Dirichlet process $DP(G_0, \alpha)$. 

Furthermore, we can easily extend this to define the generative process of a nonparametric mixture model: every table $t$ that has at least one customer seated is associated with a set of cluster parameters $\theta_t$, which were themselves drawn from some base distribution $G_0$. For each new observation, first assign that observation to a table according to the above probabilities; then, that observation is drawn from the distribution parameterized by the cluster parameters for that table. If the observation was assigned to a new table, draw a new set of cluster parameters from $G_0$, and then draw the observation from the distribution parameterized by those cluster parameters.

While this formulation of a Dirichlet process mixture model is intuitive, it is also very difficult to perform inference on in a probabilistic programming framework. This motivates an alternative formulation of DPMMs, which has empirically been shown to be more conducive to inference (e.g. Blei and Jordan, 2004).

## The Stick-Breaking Method (Sethuraman, 1994)

The generative process for the stick-breaking formulation of DPMMs proceeds as follows:

* Draw $\beta_i \sim \text{Beta}(1, \alpha)$ for $i \in \mathbb{N}$
* Draw $\theta_i \sim G_0$ for $i \in \mathbb{N}$
* Construct the mixture weights $\pi$ by taking $\pi_i(\beta_{1:\infty}) = \beta_i \prod_{j<i} (1-\beta_j)$

* For each observation $n \in \{1, ..., N\}$, draw $z_n \sim \pi(\beta_{1:\infty})$, and then draw $x_n \sim f(\theta_{z_n})$

Here, the infinite nature of the Dirichlet process mixture model can more easily be seen. Furthermore, all $\beta_i$ are independent, so it is far easier to perform inference in a probabilistic programming framework.

First, we import all the modules we're going to need:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 400
colab_type: code
id: VaAgD92IdE6i
outputId: 47dd309c-4689-403e-900b-413381c4fedc
---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
from pyro.distributions import *
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam

assert pyro.__version__.startswith('1.7.0')
pyro.set_rng_seed(0)
```

+++ {"colab_type": "text", "id": "GomHVM-cBrQY"}



## **Inference**




### Synthetic Mixture of Gaussians

We begin by demonstrating the capabilities of Dirichlet process mixture models on a synthetic dataset generated by a mixture of four 2D Gaussians:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 281
colab_type: code
id: 3_EyZAbwTMrs
outputId: f5c2222c-c3db-4a34-f021-75f38bbcb4c2
---
data = torch.cat((MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([50]),
                  MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([50]),
                  MultivariateNormal(torch.tensor([1.5, 2]), torch.eye(2)).sample([50]),
                  MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)).sample([50])))

plt.scatter(data[:, 0], data[:, 1])
plt.title("Data Samples from Mixture of 4 Gaussians")
plt.show()
N = data.shape[0]
```

+++ {"colab_type": "text", "id": "CglLQke4gEYd"}

In this example, the cluster parameters $\theta_i$ are two dimensional vectors describing the means of a multivariate Gaussian with identity covariance. Therefore, the Dirichlet process base distribution $G_0$ is also a multivariate Gaussian (i.e. the conjugate prior), although this choice is not as computationally useful, since we are not performing coordinate-ascent variational inference but rather black-box variational inference using Pyro. 

First, let's define the "stick-breaking" function that generates our weights, given our samples of $\beta$:

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: 2ngrqFlDQYpV

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
```

+++ {"colab_type": "text", "id": "LAC0bWL6Qcc3"}

Next, let's define our model. It may be helpful to refer the definition of the stick-breaking model presented in the first part of this tutorial. 

Note that all $\beta_i$ samples are conditionally independent, so we model them using a `pyro.plate` of size `T-1`; we do the same for all samples of our cluster parameters $\mu_i$. We then construct a Categorical distribution whose parameters are the mixture weights using our sampled $\beta$ values (line 9) below, and sample the cluster assignment $z_n$ for each data point from that Categorical. Finally, we sample our observations from a multivariate Gaussian distribution whose mean is exactly the cluster parameter corresponding to the assignment $z_n$ we drew for the point $x_n$. This can be seen in the Pyro code below:

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: WfnbSIocRlvQ

def model(data):
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T):
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)
```

+++ {"colab_type": "text", "id": "1gBQj5RKRn8Z"}

Now, it's time to define our guide and perform inference. 

The variational family $q(\beta, \theta, z)$ that we are optimizing over during variational inference is given by:

$$q(\beta, \theta, z) = \prod_{t=1}^{T-1} q_t(\beta_t) \prod_{t=1}^T q_t(\theta_t) \prod_{n=1}^N q_n(z_n)$$ 

Note that since we are unable to computationally model the infinite clusters posited by the model, we truncate our variational family at $T$ clusters. This does not affect our model; rather, it is a simplification made in the *inference* stage to allow tractability. 

The guide is constructed exactly according to the definition of our variational family $q(\beta, \theta, z)$ above. We have $T-1$ conditionally independent Beta distributions for each $\beta$ sampled in our model, $T$ conditionally independent multivariate Gaussians for each cluster parameter $\mu_i$, and $N$ conditionally independent Categorical distributions for each cluster assignment $z_n$.

Our variational parameters (`pyro.param`) are therefore the $T-1$ many positive scalars that parameterize the second parameter of our variational Beta distributions (the first shape parameter is fixed at $1$, as in the model definition), the $T$ many two-dimensional vectors that parameterize our variational multivariate Gaussian distributions (we do not parameterize the covariance matrices of the Gaussians, though this should be done when analyzing a real-world dataset for more flexibility), and the $N$ many $T$-dimensional vectors that parameterize our variational Categorical distributions:

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: Imw4wcVkT9er

def guide(data):
    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 3 * torch.eye(2)).sample([T]))
    phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

    with pyro.plate("beta_plate", T-1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

    with pyro.plate("mu_plate", T):
        q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2)))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))
```

+++ {"colab_type": "text", "id": "H0He1id0T_bN"}

When performing inference, we set our 'guess' for the maximum number of clusters in the dataset to $T = 6$. We define the optimization algorithm (`pyro.optim.Adam`) along with the Pyro SVI object and train the model for 1000 iterations. 

After performing inference, we construct the Bayes estimators of the means (the expected values of each factor in our variational approximation) and plot them in red on top of the original dataset.  Note that we also have we removed any clusters that have less than a certain weight assigned to them according to our learned variational distributions, and then re-normalize the weights so that they sum to one:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 372
colab_type: code
id: x1Yidukpd9wO
outputId: b0cc290b-3285-4f36-c2a6-7195b6801482
---
T = 6
optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
losses = []

def train(num_iterations):
    pyro.clear_param_store()
    for j in tqdm(range(num_iterations)):
        loss = svi.step(data)
        losses.append(loss)

def truncate(alpha, centers, weights):
    threshold = alpha**-1 / 100.
    true_centers = centers[weights > threshold]
    true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
    return true_centers, true_weights

alpha = 0.1
train(1000)

# We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
Bayes_Centers_01, Bayes_Weights_01 = truncate(alpha, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))

alpha = 1.5
train(1000)

# We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
Bayes_Centers_15, Bayes_Weights_15 = truncate(alpha, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], color="blue")
plt.scatter(Bayes_Centers_01[:, 0], Bayes_Centers_01[:, 1], color="red")

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], color="blue")
plt.scatter(Bayes_Centers_15[:, 0], Bayes_Centers_15[:, 1], color="red")
plt.tight_layout()
plt.show()
```

+++ {"colab_type": "text", "id": "DO3XDmEwC5ra"}

The plots above demonstrate the effects of the scaling hyperparameter $\alpha$. A greater $\alpha$ yields a more heavy-tailed distribution of the weights, whereas smaller $\alpha$ will place more mass on fewer clusters. In particular, the middle cluster looks like it could be generated a single Gaussian (although in fact it was generated by two distinct Gaussians), and thus the setting of $\alpha$ allows the practitioner to further encode their prior beliefs about how many clusters the data contains.

+++ {"colab_type": "text", "id": "oWBlcZ1vfiBV"}

### Dirichlet Mixture Model for Long Term Solar Observations 

As mentioned earlier, the Dirichlet process mixture model truly shines when exploring a dataset whose latent geometric structure is completely unknown. To demonstrate this, we fit a DPMM on sunspot count data taken over the past 300 years (provided by the Royal Observatory of Belgium):

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 295
colab_type: code
id: UlhV2Fx_fVGy
outputId: 045dbfaa-27a0-4aa6-ee8c-de9f9937aa38
---
df = pd.read_csv('http://www.sidc.be/silso/DATA/SN_y_tot_V2.0.csv', sep=';', names=['time', 'sunspot.year'], usecols=[0, 1])
data = torch.tensor(df['sunspot.year'].values, dtype=torch.float32).round()
N = data.shape[0]

plt.hist(df['sunspot.year'].values, bins=40)
plt.title("Number of Years vs. Sunspot Counts")
plt.xlabel("Sunspot Count")
plt.ylabel("Number of Years")
plt.show()
```

+++ {"colab_type": "text", "id": "1BE18id2_VlV"}

For this example, the cluster parameters $\theta_i$ are rate parameters since we are constructing a scale-mixture of Poisson distributions. Again, $G_0$ is chosen to be the conjugate prior, which in this case is a Gamma distribution, though this still does not strictly matter for doing inference through Pyro. Below is the implementation of the model:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 298
colab_type: code
id: f7mWbeTz_GLI
outputId: bd64a9bc-5aa2-4767-d71f-bdcb6279526c
---
def model(data):
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("lambda_plate", T):
        lmbda = pyro.sample("lambda", Gamma(3, 0.05))
    
    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", Poisson(lmbda[z]), obs=data)

def guide(data):
    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
    tau_0 = pyro.param('tau_0', lambda: Uniform(0, 5).sample([T]), constraint=constraints.positive)
    tau_1 = pyro.param('tau_1', lambda: LogNormal(-1, 1).sample([T]), constraint=constraints.positive)
    phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

    with pyro.plate("beta_plate", T-1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

    with pyro.plate("lambda_plate", T):
        q_lambda = pyro.sample("lambda", Gamma(tau_0, tau_1))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))

T = 20
alpha = 1.1
n_iter = 1500
optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
losses = []

train(n_iter)

samples = torch.arange(0, 300).type(torch.float)

tau0_optimal = pyro.param("tau_0").detach()
tau1_optimal = pyro.param("tau_1").detach()
kappa_optimal = pyro.param("kappa").detach()

# We make a point-estimate of our latent variables using the posterior means of tau and kappa for the cluster params and weights
Bayes_Rates = (tau0_optimal / tau1_optimal)
Bayes_Weights = mix_weights(1. / (1. + kappa_optimal))

def mixture_of_poisson(weights, rates, samples):
    return (weights * Poisson(rates).log_prob(samples.unsqueeze(-1)).exp()).sum(-1)

likelihood = mixture_of_poisson(Bayes_Weights, Bayes_Rates, samples)

plt.title("Number of Years vs. Sunspot Counts")
plt.hist(data.numpy(), bins=60, density=True, lw=0, alpha=0.75);
plt.plot(samples, likelihood, label="Estimated Mixture Density")
plt.legend()
plt.show()
```

+++ {"colab_type": "text", "id": "gjn4x2JqBfll"}

The above plot is the mixture density of the Bayes estimators of the cluster parameters, weighted by their corresponding weights. As in the Gaussian example, we have taken the Bayes estimators of each cluster parameter and their corresponding weights by computing the posterior means of `lambda` and `beta` respectively.

+++ {"colab_type": "text", "id": "NaHLTiRJUxBR"}

### ELBO Behavior

Below are plots of the behavior of the loss function (negative Trace_ELBO) over the SVI iterations during inference using Pyro, as well as a plot of the autocorrelations of the ELBO 'time series' versus iteration number. We can see that around 500 iterations, the loss stops decreasing significantly, so we can assume it takes around 500 iterations to achieve convergence. The autocorrelation plot reaches an autocorrelation very close to 0 around a lag of 500, further corroborating this hypothesis. Note that these are heuristics and do not necessarily imply convergence.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 369
colab_type: code
id: XzbxzSkOUveF
outputId: a752e5e8-1fcf-4ff9-d8db-8d3f59a6a32b
---
elbo_plot = plt.figure(figsize=(15, 5))

elbo_ax = elbo_plot.add_subplot(1, 2, 1)
elbo_ax.set_title("ELBO Value vs. Iteration Number for Pyro BBVI on Sunspot Data")
elbo_ax.set_ylabel("ELBO")
elbo_ax.set_xlabel("Iteration Number")
elbo_ax.plot(np.arange(n_iter), losses)

autocorr_ax = elbo_plot.add_subplot(1, 2, 2)
autocorr_ax.acorr(np.asarray(losses), detrend=lambda x: x - x.mean(), maxlags=750, usevlines=False, marker=',')
autocorr_ax.set_xlim(0, 500)
autocorr_ax.axhline(0, ls="--", c="k", lw=1)
autocorr_ax.set_title("Autocorrelation of ELBO vs. Lag for Pyro BBVI on Sunspot Data")
autocorr_ax.set_xlabel("Lag")
autocorr_ax.set_ylabel("Autocorrelation")
elbo_plot.tight_layout()
plt.show()
```

+++ {"colab_type": "text", "id": "_sn3OS8tOoko"}

## **Criticism**



+++ {"colab_type": "text", "id": "qvRAgapHZLPA"}

### Long-Term Sunspot Model

Since we computed the approximate posterior of the DPMM that was fit to the long-term sunspot data, we can utilize some intrinsic metrics, such as the log predictive, posterior dispersion indices, and posterior predictive checks. 

Since the posterior predictive distribution for a Dirichlet process mixture model is itself a scale-mixture distribution that has an analytic approximation [(Blei and Jordan, 2004)](http://www.cs.columbia.edu/~blei/papers/BleiJordan2004.pdf), this makes it particularly amenable to the aforementioned metrics:

$$p(x_{new} | X_{1:N}, \alpha, G_0) \approx \sum_{t=1}^T \mathbb{E}_q [\pi_t(\beta)] \ \mathbb{E}_q \left[p(x_{new} | \theta_t)\right].$$

In particular, to compute the log predictive, we first compute the posterior predictive distribution (defined above) after performing variational inference on our model using a training subsample of our data. The log predictive is then the log value of the predictive density evaluated at each point in the test subsample:

$$\log p(x_{new} | X) = \log \mathbb{E}_{\beta, \theta | X} \left[ p(x_{new} | \beta, \theta) \right]$$

Since both the training samples and the testing samples were taken from the same dataset, we would expect the model to assign high probability to the test samples, despite not having seen them during inference. This gives a metric by which to select values of $T$, $\alpha$, and $G_0$, our hyperparameters: we would want to choose the values that maximize this value.

We perform this process below with varying values of $\alpha$ to see what the optimal setting is.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 526
colab_type: code
id: 3V2-DC9nobPi
outputId: 473035e6-036f-4e34-9ea4-6c2dc5a4bec0
---
# Hold out 10% of our original data to test upon
df_test = df.sample(frac=0.1)
data = torch.tensor(df.drop(df_test.index)['sunspot.year'].values, dtype=torch.float).round()
data_test = torch.tensor(df_test['sunspot.year'].values, dtype=torch.float).round()
N = data.shape[0]
N_test = data_test.shape[0]

alphas = [0.05, 0.1, 0.5, 0.75, 0.9, 1., 1.25, 1.5, 2, 2.5, 3]
log_predictives = []

for val in alphas:
    alpha = val
    T = 20
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    train(500)
    
    S = 100 # number of Monte Carlo samples to use in posterior predictive computations

    # Using pyro's built in posterior predictive class:
    posterior = Predictive(guide, num_samples=S, return_sites=["beta", "lambda"])(data)
    post_pred_weights = mix_weights(posterior["beta"])
    post_pred_clusters = posterior["lambda"]

    # log_prob shape = N_test x S
    log_prob = (post_pred_weights.log() + Poisson(post_pred_clusters).log_prob(data.reshape(-1, 1, 1))).logsumexp(-1)
    mean_log_prob = log_prob.logsumexp(-1) - np.log(S)
    log_posterior_predictive = mean_log_prob.sum(-1)
    log_predictives.append(log_posterior_predictive)

plt.figure(figsize=(10, 5))
plt.plot(alphas, log_predictives)
plt.title("Value of the Log Predictive at Varying Alpha")
plt.show()
```

+++ {"colab_type": "text", "id": "PGw4eD363PWP"}

From the above plot, we would surmise that we want to set $\alpha > 1$, though the signal is not quite clear. A more comprehensive model criticism process would involve performing a grid search across all hyperparameters in order to find the one that maximizes the log predictive.

+++ {"colab_type": "text", "id": "8A1gt4sn8Upo"}

## References



1.   Ferguson, Thomas. *A Bayesian Analysis of Some Nonparametric Problems*. The Annals of Statistics, Vol. 1, No. 2 (1973). 
2.   Aldous, D. *Exchangeability and Related Topics*. Ecole diete de Probabilities Saint Flour (1985).
3.   Sethuraman, J. *A Constructive Definition of Dirichlet Priors*. Statistica, Sinica, 4:639-650 (1994).
4.   Blei, David and Jordan, Michael. *Variational Inference for Dirichlet Process Mixtures*. Bayesian Analysis, Vol. 1, No. 1 (2004).
5.   Pedregosa, et al. *Scikit-Learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830 (2011).
6.   Bishop, Christopher. *Pattern Recogition and Machine Learning*. Springer Ltd (2006).
7.   *Sunspot Index and Long-Term Solar Observations*. WDC-SILSO, Royal Observatory of Belgium, Brussels (2018).
9.   Gelman, Andrew. *Understanding predictive information criteria for Bayesian models*. Statistics and Computing, Springer Link, 2014.
