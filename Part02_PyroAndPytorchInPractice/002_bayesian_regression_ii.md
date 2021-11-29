---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 贝叶斯回归 - 推断算法（第二部分）

+++

在 [第一部分](001_bayesian_regression.ipynb) 中，我们研究了如何使用 SVI 对简单的贝叶斯线性回归模型进行推断。在本教程中，我们将探索更具表现力的`引导函数`以及更为精确的推断技术。

我们将使用与以前相同的数据集。

```{code-cell} ipython3
%reset -sf
```

```{code-cell} ipython3
import logging
import os

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

pyro.set_rng_seed(1)
assert pyro.__version__.startswith('1.7.0')
```

```{code-cell} ipython3
%matplotlib inline
plt.style.use('default')

logging.basicConfig(format='%(message)s', level=logging.INFO)
smoke_test = ('CI' in os.environ)
pyro.set_rng_seed(1)
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
rugged_data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
```

## 1 目标

再次重申下目标：根据数据集中的两个特征预测一个国家的对数人均 GDP，但这次将探索更具表现力的`引导函数`。

+++

## 2 模型与引导（ `Model`  and `Guide` ）

我们将再次写出类似于 [第一部分](001_bayesian_regression.ipynb) 中的模型，但不使用 `PyroModule`。我们将使用相同的先验写出回归中的每一项。 `bA` 和 `bR` 是对应于 `is_cont_africa` 和 `ruggedness` 的回归系数，`a` 是截距，`bAR` 是两个特征之间的相关因子。

编写 `guide` 的过程与构建 `model` 的过程非常相似，主要区别在于`guide` 的参数必须是可训练的。为此，使用 `pyro.param()` 在参数存储库中注册 `guide` 的参数。注意示例中，各 `scale` 参数的正实值约束条件。

```{code-cell} ipython3
def model(is_cont_africa, ruggedness, log_gdp):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
    with pyro.plate("data", len(ruggedness)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)
        
def guide(is_cont_africa, ruggedness, log_gdp):
    a_loc = pyro.param('a_loc', torch.tensor(0.))
    a_scale = pyro.param('a_scale', torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                             constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', torch.randn(3))
    weights_scale = pyro.param('weights_scale', torch.ones(3),
                               constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness
```

```{code-cell} ipython3
# Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

# Prepare training data
df = rugged_data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])
train = torch.tensor(df.values, dtype=torch.float)
```

## 3 随机变分推断

和之前一样，使用随机变分推断来执行推断任务。

```{code-cell} ipython3
from pyro.infer import SVI, Trace_ELBO


svi = SVI(model, 
          guide, 
          optim.Adam({"lr": .05}), 
          loss=Trace_ELBO())

is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]
pyro.clear_param_store()
num_iters = 5000 if not smoke_test else 2
for i in range(num_iters):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    if i % 500 == 0:
        logging.info("Elbo loss: {}".format(elbo))
```

```{code-cell} ipython3
from pyro.infer import Predictive


num_samples = 1000
predictive = Predictive(model, guide=guide, num_samples=num_samples)
svi_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
               for k, v in predictive(log_gdp, is_cont_africa, ruggedness).items()
               if k != "obs"}
```

让我们观察模型中不同隐变量的后验分布。

```{code-cell} ipython3
for site, values in summary(svi_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")
```

## 4 汉密尔顿蒙特卡洛（HMC）推断

与使用变分推断获得隐变量的近似后验相比，我们还可以使用 [马尔可夫链蒙特卡罗](http://docs.pyro.ai/en/dev/mcmc.html) ( MCMC）推断，一大类允许我们从真实后验中抽取无偏样本的算法。

本例中将使用的 MCMC 算法为 `No-U Turn Sampler (NUTS)` [1]，它提供了一种高效运行汉密尔顿蒙特卡洛的自动化方法。虽然它比变分推断稍慢，但能够提供更精确的估计。

> 注： 当后验分布无法获得解析解时，通常会利用 MCMC 在极限情况下趋近于真实分布的优势，将其推断结果作为真实后验的黄金标准。

```{code-cell} ipython3
from pyro.infer import MCMC, NUTS

nuts_kernel = NUTS(model)

mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
mcmc.run(is_cont_africa, ruggedness, log_gdp)

hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
```

```{code-cell} ipython3
for site, values in summary(hmc_samples).items():
    print("Site: {}".format(site))
    print(values, "\n")
```

## 5 两者的比较

让我们比较一下两者的推断结果。如下所示，对于变分推断，不同隐变量的边缘分布相对于真实后验（来自 HMC ）过于聚集了（即过于自信了）。当我们绘制来自变分推断的联合后验分布的不同剖面（即隐变量的边缘分布）时，可以更直观地看到这一点。上例中，由于模型选用了具有对角协方差矩阵的变分族，因此无法对隐变量之间的任何相关性建模，并且产生了过于自信的近似。

```{code-cell} ipython3
sites = ["a", "bA", "bR", "bAR", "sigma"]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    site = sites[i]
    sns.distplot(svi_samples[site], ax=ax, label="SVI (DiagNormal)")
    sns.distplot(hmc_samples[site], ax=ax, label="HMC")
    ax.set_title(site)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');
```

```{code-cell} ipython3
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Cross-section of the Posterior Distribution", fontsize=16)
sns.kdeplot(hmc_samples["bA"], hmc_samples["bR"], ax=axs[0], shade=True, label="HMC")
sns.kdeplot(svi_samples["bA"], svi_samples["bR"], ax=axs[0], label="SVI (DiagNormal)")
axs[0].set(xlabel="bA", ylabel="bR", xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
sns.kdeplot(hmc_samples["bR"], hmc_samples["bAR"], ax=axs[1], shade=True, label="HMC")
sns.kdeplot(svi_samples["bR"], svi_samples["bAR"], ax=axs[1], label="SVI (DiagNormal)")
axs[1].set(xlabel="bR", ylabel="bAR", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');
```

## 6 增强引导的相关性建模能力

与之前对角型方差矩阵的多元正态引导函数不同，我们现在将使用一个能够对相关性建模的新引导。该引导从多元正态分布的 Cholesky 分解生成样本，从而使我们能够通过协方差矩阵捕获隐变量之间的相关性。

如果手动编写代码，将需要组合所有隐变量以便可以从多元正态分布中联合采样。

```{code-cell} ipython3
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean

guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)

svi = SVI(model, 
          guide, 
          optim.Adam({"lr": .01}), 
          loss=Trace_ELBO())

is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]
pyro.clear_param_store()
for i in range(num_iters):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    if i % 500 == 0:
        logging.info("Elbo loss: {}".format(elbo))
```

此后再看一下后验的形状，可以看到新引导函数能够捕获更多真实后验的信息。

```{code-cell} ipython3
predictive = Predictive(model, guide=guide, num_samples=num_samples)
svi_mvn_samples = {k: v.reshape(num_samples).detach().cpu().numpy() 
                   for k, v in predictive(log_gdp, is_cont_africa, ruggedness).items()
                   if k != "obs"}
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.suptitle("Marginal Posterior density - Regression Coefficients", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    site = sites[i]
    sns.distplot(svi_mvn_samples[site], ax=ax, label="SVI (Multivariate Normal)")
    sns.distplot(hmc_samples[site], ax=ax, label="HMC")
    ax.set_title(site)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');
```

现在比较一下两种不同引导函数计算的后验结果。可以发现，后者比前者更发散，与真实后验更接近，也就是说其生成的近似后验不会像前者那么自信。

```{code-cell} ipython3
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Cross-sections of the Posterior Distribution", fontsize=16)
sns.kdeplot(svi_samples["bA"], svi_samples["bR"], ax=axs[0], label="SVI (Diagonal Normal)")
sns.kdeplot(svi_mvn_samples["bA"], svi_mvn_samples["bR"], ax=axs[0], shade=True, label="SVI (Multivariate Normal)")
axs[0].set(xlabel="bA", ylabel="bR", xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
sns.kdeplot(svi_samples["bR"], svi_samples["bAR"], ax=axs[1], label="SVI (Diagonal Normal)")
sns.kdeplot(svi_mvn_samples["bR"], svi_mvn_samples["bAR"], ax=axs[1], shade=True, label="SVI (Multivariate Normal)")
axs[1].set(xlabel="bR", ylabel="bAR", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');
```

以下是第二种`引导`的推断结果与 HMC 方法结果进行比较，可以看到新`引导` 能够更好地捕捉真实后验的信息。

```{code-cell} ipython3
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Cross-sections of the Posterior Distribution", fontsize=16)
sns.kdeplot(hmc_samples["bA"], hmc_samples["bR"], ax=axs[0], shade=True, label="HMC")
sns.kdeplot(svi_mvn_samples["bA"], svi_mvn_samples["bR"], ax=axs[0], label="SVI (Multivariate Normal)")
axs[0].set(xlabel="bA", ylabel="bR", xlim=(-2.5, -1.2), ylim=(-0.5, 0.1))
sns.kdeplot(hmc_samples["bR"], hmc_samples["bAR"], ax=axs[1], shade=True, label="HMC")
sns.kdeplot(svi_mvn_samples["bR"], svi_mvn_samples["bAR"], ax=axs[1], label="SVI (Multivariate Normal)")
axs[1].set(xlabel="bR", ylabel="bAR", xlim=(-0.45, 0.05), ylim=(-0.15, 0.8))
handles, labels = axs[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right');
```

## 参考文献
[1] Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning Research 15.1 (2014): 1593-1623.  https://arxiv.org/abs/1111.4246.
