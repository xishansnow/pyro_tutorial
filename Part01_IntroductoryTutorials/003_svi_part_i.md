---
jupytext:
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

# 随机变分推断 （ $\mathrm{I}$ ）:  Pyro 中的随机变分推断（SVI）

Pyro 支持很多种推断方法，但更多专注于随机变分推断（SVI）算法。让我们看看如何在 Pyro 中做变分推断。

## 1 基本设置

假设已经在 Pyro 中定义了一个模型，方法见 [介绍（ I ）](001_intro_part_i.ipynb) 。

我们先作一下快速回顾，该模型是作为随机函数的形式 `model(*args, **kwargs)` 给出的，其中带有若干参数。不同的 `model()`  通常通过以下映射被编码：

（1） 观测数据 $\Longleftrightarrow$  `pyro.sample` 的 `obs` 参数，或 `pyro.condition` 元语。<br>
（2） 隐变量  $\Longleftrightarrow$  `pyro.sample`。 <br>
（3） 参数 $\Longleftrightarrow$ `pyro.param`。 <br>

首先建立一些基本概念。 

一般性的，`模型` 包含观测数据  ${\bf x}$ 、隐变量 ${\bf z}$ 以及参数  $\boldsymbol{\theta}$ 。模型具有如下形式的联合概率密度：

$$
p_{\theta}({\bf x}, {\bf z}) = p_{\theta}({\bf x}|{\bf z}) p_{\theta}({\bf z})
$$

假设构成 $p_{\theta}({\bf x}, {\bf z})$ 的各概率分布 $p_i$ 具有如下属性：

> 注：此处 $p_i$ 应当是指 $p_{\theta}({\bf x}|{\bf z})$ 和 $p_{\theta}({\bf z})$ 。

（1） 我们能够在各 $p_i$ 中进行采样。<br>
（2） 我们能够逐点计算 $p_i$ 的（对数）概率密度值。<br>
（3） $p_i$ 相对于参数 $\theta$ 可微 --- 优化的要求。 <br>


## 2 模型学习

当前判断是否学习了一个好模型的准则主要是**最大化对数证据**，即我们希望找到满足如下条件的 $\theta$ 值：

$$
\theta_{\rm{max}} = \underset{\theta}{\operatorname{arg max}} \log p_{\theta}({\bf x})
$$

其中，对数证据  $\log p_{\theta}({\bf x})$ 通过下式的边缘化获得：

$$
\log p_{\theta}({\bf x}) = \log \int p_{\theta}({\bf x}, {\bf z})  d{\bf z}
$$

通常情况下，这是一个非常困难的问题。主要是因为隐变量 $\bf z$ 上的积分非常棘手，即使 $\theta$ 是固定值也很难计算。更甚者，即使我们知道如何为每个 $\theta$ 值计算其对数证据，将最大化对数证据作为优化 $\theta$ 的目标函数通常也是一个难度很大的非凸优化问题。

除了找到 $\theta_{\rm{max}}$ 之外，我们还想计算潜在变量 $\bf z$ 的后验：

$$
 p_{\theta_{\rm{max}}}({\bf z} | {\bf x}) = \frac{p_{\theta_{\rm{max}}}({\bf x} , {\bf z})}{
\int p_{\theta_{\rm{max}}}({\bf x} , {\bf z}) d{\bf z} } 
$$

注意，该表达式的分母就是通常比较棘手的证据，变分推断就是为了找到 $\theta_{\rm{max}}$ 的解，并且计算后验分布  $p_{\theta_{\rm{max}}}({\bf z} | {\bf x})$ 的近似分布。

让我们看一下它是如何工作的。

## 3 引导函数（变分分布）

变分推断的基本想法是引入一个参数化的分布 $q_{\phi}({\bf z})$ ，其中 $\phi$ 被称为变分参数，而 $q_{\phi}({\bf z})$ 习惯被称为变分分布。在 Pyro 环境中，为称呼方便，变分分布被称为引导函数（或简称为引导）。当后验分布非常复杂、没有解析形式解的情况下，引导函数（变分分布）将作为真实后验分布的近似：

>  注：专业术语均被成为变分分布，只有 Pyro 中被称为引导函数。介绍资料中解释说，主要是因为英文中变分分布（Variational Distribution）这几个字的音节数太多，发音比较麻烦 ，而引导函数（Guide）只需要一个音节。


就像`模型`被编码为一个随机函数 `model()` 一样，`引导`也被编码为一个随机函数 `guide()`。其中包含 `pyro.sample` 和 `pyro.param` 语句，但不包含观测数据 $\mathbf{x}$。

> **注意：**
> Pyro 强制要求 `model()` 和`guide()` 具有相同的调用接口形式，即这两个可调用对象应该采用相同的参数。

引导是对后验 $p_{\theta_{\rm{max}}}({\bf z} | {\bf x})$ 的近似，因此它需要提供一个在所有隐变量上的有效联合概率密度。回想一下，当在 Pyro 中使用元语 `pyro.sample()` 指定随机变量时，第一个参数表示随机变量的名称。这些随机变量的名称将被用于对齐`模型`和`引导`中的随机变量。非常明确，如果`模型`中包含一个随机变量 `z_1` ，那么`引导`中需要具备一个匹配的 `sample` 语句：

```python
def model():
    pyro.sample("z_1", ...)

def guide():
    pyro.sample("z_1", ...)
```

上述 `model()` 和 `guide()` 中的分布可以不同，但随机变量的名称必须一一对应。 

一旦指定了引导，我们就可以实施推理了。学习将被设置为一个优化问题，其中每次训练迭代都在 $\theta-\phi$ 空间中执行一步，以使引导函数更接近真实后验。为此，需要定义一个适当的目标函数。

## 4 证据下界 ELBO

一个简单的推导（例如参见参考文献 [1]）就可以生成我们想要的证据下限  ( ELBO )。 ELBO 是 $\theta$ 和 $\phi$ 的函数，被定义为关于引导函数样本的期望：

$$
{\rm ELBO} \equiv \mathbb{E}_{q_{\phi}({\bf z})} \left [ 
\log p_{\theta}({\bf x}, {\bf z}) - \log q_{\phi}({\bf z})
\right]
$$

假设我们可以计算期望里面的对数概率。由于引导被定义为一个能够从中采样的参数化分布，因此我们可以基于 $q_\phi(\mathbf{z})$ 的样本来计算期望的蒙特卡罗估计。

至关重要的是，ELBO 是对数证据的下界，即对于 $\theta$ 和 $\phi$ 的所有选择，有：

$$
\log p_{\theta}({\bf x}) \ge {\rm ELBO} 
$$

因此，如果采取（随机的）梯度步骤来最大化 ELBO ，我们也会将对数证据推高（在期望值上）。此外，可以证明 ELBO 和对数证据之间的差，就是`引导`和`真实后验分布`之间的 KL 散度：

$$
 \log p_{\theta}({\bf x}) - {\rm ELBO} = 
\rm{KL}\!\left( q_{\phi}({\bf z}) \lVert p_{\theta}({\bf z} | {\bf x}) \right) 
$$

这个 KL 散度是两个分布之间“接近度”的特定（非负）度量。因此，对于固定的 $\theta$，当我们在 $\phi$ 的参数空间中执行增加 ELBO 的步骤时，会减少`引导`和`后验分布`之间的 KL 散度，也就是说，`引导`移向`后验分布`方向。而对于固定的 $\phi$ ， 在 $\theta$ 的参数空间中执行增加对数证据的步骤，会增加`引导`和`后验分布`之间的 KL 散度，但更倾向于一个更好的模型。

在一般情况下，我们在 $\theta$ 和 $\phi$ 参数空间中同时采取梯度步骤，以便引导能够跟踪动态优化中的后验分布 $\log p_{\theta}({\bf z } | {\bf x})$ 。这也许令人有些惊讶，目标在变动的同时，`引导`需要逐步靠近它。 好在事实表明：对于许多问题，此优化是有解的。

所以，在顶层理解变分推断很容易：我们需要做的就是定义一个引导并计算 ELBO 的梯度。

但实际上，计算`模型`和`引导`对之间的梯度会导致一些非常复杂情况（有关讨论，请参阅教程 [SVI Part III](005_svi_part_iii.ipynb)）。此教程仅考虑一个已解决的问题，以便了解 Pyro 为变分推断提供的支持。

## 5 `Pyro` 中的 `SVI` 类

在 Pyro 中，执行变分推断的机器被分装在 `SVI` 类中。

为了执行变分推断，用户需要向 `SVI` 类提供三样东西：**模型、引导函数和一个优化器**。其中`模型`和`引导`已经讨论过，下面会重点了解优化器。

假设用户已经有了上述三样东西，为了构建执行 ELBO 目标优化的 `SVI` 实例，可以编写：

```python
import pyro
from pyro.infer import SVI, Trace_ELBO
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
```

 `SVI` 对象提供两个方法： `step()` 和 `evaluate_loss()` ，用于封装变分学习和变分评估的逻辑：

(1)  `step()` 方法执行一次梯度步骤，并返回对损失的估计。如果提供了`step()` 的参数，则它将通过管道传送到 `model()` 和 `guide()`。<br>

(2)  `evaluate_loss()` 方法返回一个未执行梯度步骤的损失估计。就像 `step()` 一样， `evaluate_loss()` 的参数将通过管道传送给 `model()`和 `guide()` 。<br>

当损失函数为 ELBO 时，两种方法也都接受一个可选的参数  `num_particles`，该参数表示用于计算损失和梯度的样本数量。根据上面的两条说明，`step()`同时包含损失和梯度的计算，而 `evaluate_loss()` 仅包含损失的计算。

## 6 `Pyro` 中的优化器

在 Pyro 中，允许`模型`和`引导`是任意的随机函数，只要满足：

(1)  `guide` 中不包含带  `obs` 参数的 `pyro.sample` 语句，即引导与观测数据无关。<br>
(2)  `model` 和 `guide` 具有相同的调用接口形式。<br>

这带来了一些挑战，因为这意味着运行不同的 `model()` 和 `guide()` 可能会有不同的行为表现。例如：某些隐变量和参数只是偶尔出现。

实际上，参数有可能在推断过程中被创建出来。 换句话说，我们执行优化的、被 $\theta$ 和 $\phi$ 参数化的空间，可以动态的增长和变化。

> 阅读体验：此处暂时没有想到合适的案例。

为了适应这种情况， Pyro 提供了一个在学习过程中，动态配置参数的机制，即在每个参数初次出现时，动态为其生成一个优化器。幸运的是， PyTorch 提供了一个轻量级优化库（ 见 [torch.optim](http://pytorch.org/docs/master/optim.html) ），对于此类动态情况能够很容易地应对。

所有这些均由 `pyro.optim.PyroOptim` 类来控制， 该类是围绕 PyTorch 优化器的一个瘦封装。`PyroOptim` 采用两个参数： 一个 PyTorch 优化器的构造器 `optim_constructor` ，和一个优化器参数 `optim_args` 的声明。从顶层来看，在优化过程中，每当看到一个需要优化的新参数时，带有 `optim_args` 指定参数的构造器 `optim_constructor` 就被用来实例化一个指定的新优化器。

很多用户有可能不会直接和 `PyroOptim` 打交道，而是与 `optim/__init__.py` 中定义的别名打交道。让我们看看是怎么回事。

有两种途径来声明优化器的参数。

（1）第一种途径适用于稍简单的应用，此时 `optim_args` 是一个（为所有待优化参数）创建优化器时所需优化器参数的固定字典：

```python
from pyro.optim import Adam

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)
```

（2）第二种途径允许对优化器参数的更好控制。此时用户必须指定一个可调用对象，该可调用对象在为新的待优化参数创建优化器时，将被 Pyro 调用。此可调用对象必须具有以下接口形式：

① `module_name`:  那些包含待优化参数的 Pyro 模块名称（ 如果有的化）
② `param_name`:  那些待优化参数的 Pyro 名称

这给用户为不同参数定制优化器的能力（如：有差别的学习率）。如果想学习如何利用控制能力，请参见 [基线的讨论](005_svi_part_iii.ipynb) 。这里有一个简单例子来展示该 API：

```python
from pyro.optim import Adam

def per_param_callable(param_name):
    if param_name == 'my_special_parameter':
        return {"lr": 0.010}
    else:
        return {"lr": 0.001}

optimizer = Adam(per_param_callable)
```

上例简单地通知 Pyro ， 在创建 `Adam` 优化器时，为待优化参数 `my_special_parameter` 使用 `0.010` 的学习率，为其他待优化参数使用学习率 `0.001` 。

## 7 一个简单示例

我们以一个简单的例子结束。你现在有一个双面硬币，想确定硬币是否公平，即它是否以相同频率出现正面或反面。根据以下两个观察，您对硬币的公平性可能有一个先验信念：

- 这是美国铸币局发行的标准 0.25 元硬币

- 这枚硬币在使用多年后有点破损

因此，虽然你认为硬币在生产时非常公平，但也允许其公平性偏离了完美的 1:1 。因此，如果结果表明硬币以 11:10 的比例偏向于正面，您也不会感到惊讶。相比之下，如果结果表明硬币以 5:1 的比例偏向正面，反而会让你感到非常惊讶。

为了把上述陈述转变成一个概率模型，我们将正面和反面分别编码为 `1` 和 `0` 。我们将硬币的公平性编码为实数 $f$，其中 $f$ 满足 $f \in [0.0, 1.0]$ ， $f=0.50$ 对应于完全公平的硬币。

我们对 $f$ 的先验信念将通过贝塔分布进行编码，特别是 $\rm{Beta}(10,10)$，它是区间 $[0.0, 1.0]$ 上的对称概率分布，在 $f=0.5$ 处达到峰值。

```{raw-cell}
:raw_mimetype: text/html

<center><figure><img src="_static/img/beta.png" style="width: 300px;"><figcaption> <font size="-1"><b>Figure 1</b>: The distribution Beta that encodes our prior belief about the fairness of the coin. </font></figcaption></figure></center>
```

为了得到先验更准确的公平性，我们需要做实验并收集一些数据。假设抛硬币 10 次，将每次抛硬币的结果记录在列表 `data` 中，则相应的模型由下式给出：

```python
import pyro.distributions as dist

def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli 
        # likelihood Bernoulli(f)
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])
```

这里有一个隐变量  (`latent_fairness` )，它服从分布 $\rm{Beta}(10, 10)$ 。以该随机变量为条件，我们使用伯努利似然来观察每个数据点。注意，每个观察值在 Pyro 中被分配了唯一的名称。

下一个任务是定义相应的引导，即隐变量 $f$ 的变分分布。这里唯一的要求是 $q(f)$ 应该是 $[0.0, 1.0]$ 范围内的概率分布，因为 $f$ 在该范围之外没有任何意义。一个简单的选择是使用由 $\alpha_q$ 和 $\beta_q$ 参数化的另一个贝塔分布。实际上，在这种特殊情况下，这是一种 “正确” 的选择，因为伯努利分布和贝塔分布是共轭的，这意味着后验分布也是具有解析解的贝塔分布。在 Pyro 中，我们写：

```python
def guide(data):
    # register the two variational parameters with Pyro.
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0), 
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0), 
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))
```
这里有几点需要注意：

- 我们已经注意随机变量的名称在`模型`和`引导`之间完全对齐。

- `model(data)` 和 `guide(data)` 采用相同的接口参数。 

- 变分参数都表示为 `torch.tensor` 张量，其梯度计算标志 `requires_grad` 被 `pyro.param` 自动设置为 `True`。

- 使用了 `constraint=constraints.positive` 来确保 `alpha_q` 和 `beta_q` 在优化过程中保持非负。

现在继续进行随机变分推断：

```python
# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 5000
# do gradient steps
for step in range(n_steps):
    svi.step(data)
```    

注意，在 `step()` 方法中，我们传入数据，然后将其传递给`模型`和`引导`。

目前唯一缺少的是数据集。让我们创建一些数据并和上面的代码片段组合成一个完整脚本：

```{code-cell} ipython3
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
n_steps = 2 if smoke_test else 2000

assert pyro.__version__.startswith('1.7.0')

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))

def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli likelihood
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

def guide(data):
    # register the two inferred_mean = alpha_q / (alpha_q + beta_q)variational parameters with Pyro
    # - both parameters will have initial value 15.0. 
    # - because we invoke constraints.positive, the optimizer 
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0), 
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0), 
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print('.', end='')

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
```


可以将上述代码的估计值和精确后验均值（本例中为 $16/30 = 0.53$ ）做一比较。请注意，硬币公平性的最终估计介于先验（ 即 0.50 ）和经验频率建议的公平性（ 6/10 = 0.60 ）之间。

+++

## References

[1] `Automated Variational Inference in Probabilistic Programming`,
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
David Wingate, Theo Weber

[2] `Black Box Variational Inference`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Rajesh Ranganath, Sean Gerrish, David M. Blei

[3] `Auto-Encoding Variational Bayes`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Diederik P Kingma, Max Welling
