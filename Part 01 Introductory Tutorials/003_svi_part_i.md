---
jupytext:
  formats: ipynb,md:myst
  text_representation: {extension: .md, format_name: myst, format_version: 0.13, jupytext_version: 1.13.1}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# 随机变分推断 （ $\mathrm{I}$ ）:  Pyro 中的随机变分推断（SVI）

Pyro 被设计为更多专注于将随机变分推断（SVI）作为通用推断算法。向让我们在 Pyro 中如何做变分推断。

##  1 设置

假设我们已经在 Pyro 中定义了一个模型，方法见 [介绍（ I ）](intro_part_i.ipynb) 。作为快速回顾，该模型是作为随机函数 `model(*args, **kwargs)` 的形式给出的，其中带有若干参数。不同的 `model()`  代码通常通过以下映射被编码：

1.  观测数据 $\Longleftrightarrow$  `pyro.sample` 的 `obs` 参数
2.  隐变量  $\Longleftrightarrow$  `pyro.sample`
3.  参数 $\Longleftrightarrow$ `pyro.param`

首先建立一些基本概念。 模型包含观测数据  ${\bf x}$ 、隐变量 ${\bf z}$ 以及参数  $\theta$ 。模型具有如下形式的联合概率密度：

$$
p_{\theta}({\bf x}, {\bf z}) = p_{\theta}({\bf x}|{\bf z}) p_{\theta}({\bf z})
$$

假设构成 $p_{\theta}({\bf x}, {\bf z})$ 的各概率分布 $p_i$ 具有如下属性：

> 注：此处 $p_i$ 应当是指 $p_{\theta}({\bf x}|{\bf z})$ 和 $p_{\theta}({\bf z})$ 。

1.  我们能够从各 $p_i$ 中采样
2.  我们能够逐点计算对数概率密度函数值 $p_i$ 
3. $p_i$ 相对于参数 $\theta$ 可微


## 2 模型学习

当前判断是否学习了一个好模型的准则主要是最大化对数证据，即我们希望找到满足如下条件的 $\theta$ 值：

$$
\theta_{\rm{max}} = \underset{\theta}{\operatorname{arg max}} \log p_{\theta}({\bf x})
$$

其中，对数证据  $\log p_{\theta}({\bf x})$ 通过下式的边缘化获得：

$$
\log p_{\theta}({\bf x}) = \log \int\! d{\bf z}\; p_{\theta}({\bf x}, {\bf z})
$$

通常情况下，这是一个非常困难的问题。主要是因为隐变量 $\bf z$ 上的积分非常棘手，即使 $\theta$ 是个固定值也很难计算。更甚者，即使我们知道如何为每个 $\theta$ 值计算对数证据，将最大化对数证据作为优化 $\theta$ 的目标函数通常也是一个难度很大的非凸优化问题。

In addition to finding $\theta_{\rm{max}}$, we would like to calculate the posterior over the latent variables $\bf z$:

除了找到 $\theta_{\rm{max}}$ 之外，我们还想计算潜在变量 $\bf z$ 的后验：

$$
 p_{\theta_{\rm{max}}}({\bf z} | {\bf x}) = \frac{p_{\theta_{\rm{max}}}({\bf x} , {\bf z})}{
\int \! d{\bf z}\; p_{\theta_{\rm{max}}}({\bf x} , {\bf z}) } 
$$

注意，该表达式的分母就是通常比较棘手的证据，变分推断就是为了找到 $\theta_{\rm{max}}$ 的解，并且计算后验分布  $p_{\theta_{\rm{max}}}({\bf z} | {\bf x})$ 的近似分布。

让我们看一下它是如何工作的。

## 3 引导函数

变分推断的基本想法是引入一个参数化的分布 $q_{\phi}({\bf z})$ ，其中 $\phi$ 常被成为变分参数。该分布习惯被称为变分分布，而在 Pyro 环境中，它被成为引导函数。引导函数将作为后验分布的近似：

>  注：专业术语均被成为变分分布，只有 Pyro 中被称为引导函数。介绍资料中解释说，主要是因为英文中变分分布（Variational Distribution）这几个字的音节数太多，发音比较麻烦 ，而引导函数（Guide）只需要一个音节。


就像模型一样，引导函数也被编码为一个随机函数 `guide()`，其中包含 `pyro.sample` 和 `pyro.param` 语句。引导函数不包含观测数据 $\mathbf{x}$，因为该引导函数需要是某个归一化的分布。请注意，Pyro 强制 `model()` 和`guide()` 具有相同的调用签名，即两个可调用对象应该采用相同的参数。

由于引导函数是对后验 $p_{\theta_{\rm{max}}}({\bf z} | {\bf x})$ 的近似，所以它需要提供一个在所有隐变量上的有效联合概率密度。回想一下，当在 Pyro 中使用元语 `pyro.sample()` 指定随机变量时，第一个参数表示随机变量的名称。这些随机变量的名称将被用于对齐模型和引导函数中的随机变量。非常明确，如果模型包含一个随机变量 `z_1` ：

```python
def model():
    pyro.sample("z_1", ...)
```

那么引导函数需要具备一个匹配的 `sample` 语句：

```python
def guide():
    pyro.sample("z_1", ...)
```

上述 `model()` 和 `guide()` 中的分布可以不同，但随机变量的名称必须一一对应。 

一旦指定了一个引导函数，我们可以进行推理了。学习将被设置为一个优化问题，其中每次训练迭代都在 $\theta-\phi$ 空间中执行一步，以使引导函数更接近真实后验。为此，需要定义一个适当的目标函数。

## 4 证据下界 ELBO

一个简单的推导（例如参见参考文献 [1]）就可以生成我们想要的证据下限  (ELBO )。 ELBO 是 $\theta$ 和 $\phi$ 的函数，被定义为关于引导函数样本的期望：

$$
{\rm ELBO} \equiv \mathbb{E}_{q_{\phi}({\bf z})} \left [ 
\log p_{\theta}({\bf x}, {\bf z}) - \log q_{\phi}({\bf z})
\right]
$$

假设我们可以计算期望里面的对数概率。由于引导函数被假定为一个能够从中采样的参数化分布，因此我们可以计算该量的蒙特卡罗估计。至关重要的是，ELBO 是对数证据的下界，即对于 $\theta$ 和 $\phi$ 的所有选择，有：

$$
\log p_{\theta}({\bf x}) \ge {\rm ELBO} 
$$

因此，如果想采用（随机）梯度步骤来最大化 ELBO，我们也会将对数证据推得更高（在期望中）。此外，可以证明 ELBO 和对数证据之间的差，就是引导函数和真实后验之间的 KL 散度：

$$
 \log p_{\theta}({\bf x}) - {\rm ELBO} = 
\rm{KL}\!\left( q_{\phi}({\bf z}) \lVert p_{\theta}({\bf z} | {\bf x}) \right) 
$$

这个 KL 散度是两个分布之间“接近度”的特定（非负）度量。因此，对于固定的 $\theta$，当我们在 $\phi$ 的参数空间中执行增加 ELBO 的步骤时，会减少了引导函数和后验分布之间的 KL 散度，也就是说，引导函数移向后验分布方向。在一般情况下，我们在 $\theta$ 和 $\phi$ 参数空间中同时采取梯度步骤，以便引导函数能够跟踪移动的后验分布 $\log p_{\theta}({\bf z } | {\bf x})$ 。也许有点令人惊讶，尽管目标在移动，但对于许多问题，此优化是有解的。

所以，在顶层理解变分推断很容易：我们需要做的就是定义一个引导函数并计算 ELBO 的梯度。但实际上，计算模型和引导函数对之间的梯度会导致一些非常复杂情况（有关讨论，请参阅教程 [SVI Part III](005_svi_part_iii.ipynb)）。此教程仅考虑一个已解决的问题，以便了解 Pyro 为变分推断提供的支持。

## 5 `SVI` 类

在 Pyro 中，执行变分推断的机器被分装在 `SVI` 类中。

用户需要提供三个东西：模型、引导函数和一个优化器。我们已经讨论过模型和引导函数，下面会重点讨论优化器。假设已经有了上述三个组分，为了构建执行 ELBO 目标优化的 `SVI` 实例，用户编写：

```python
import pyro
from pyro.infer import SVI, Trace_ELBO
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
```

 `SVI` 对象提供两个方法： `step()` 和 `evaluate_loss()` ，用于封装变分学习和变分评估的逻辑：

1.  `step()` 方法执行一个梯度步骤，并返回损失的估计。如果提供了`step()` 的参数，则它将通过管道传送到 `model()` 和 `guide()`。

2.  `evaluate_loss()` 方法返回一个未执行梯度步骤的损失估计。就像 `step()` 一样， `evaluate_loss()` 的参数将通过管道传送给 `model()`和 `guide()` 。

当损失函数为 ELBO 时，两种方法也都接受一个可选的参数  `num_particles`，该参数表示用于计算损失的样本数量 (在`evaluate_loss` 方法中) 以及损失和梯度 (在 `step` 方法中). 

## 6 优化器

在 Pyro 中，允许模型和引导函数是任意的随机函数，只要满足：

1.  `guide` 中不包含带  `obs` 参数的 `pyro.sample` 语句
2.  `model` 和 `guide` 具有相同的调用签名

这带来了一些挑战，因为这意味着运行不同的 `model()` 和 `guide()` 可能会有非常不同的行为表现，例如某些隐变量和参数只是偶尔出现。实际上，参数有可能在推断过程中被创建出来。 换句话说，我们执行优化的、被 $\theta$ 和 $\phi$ 参数化的空间，可以动态的增长和变化。

为了适应这种情况， Pyro 需要在学习过程中，在每个参数初次出现时，动态为其生成一个优化器。幸运的是， PyTorch 有一个轻量级的优化库（ 见 [torch.optim](http://pytorch.org/docs/master/optim.html) ） ，对于动态情况能够很容易地被重新利用。

所有这些均由 `optim.PyroOptim` 类来控制， 该类是围绕 PyTorch 优化器的一个瘦封装。`PyroOptim` 采用两个参数： 一个 PyTorch 优化器的构造器 `optim_constructor` ，和一个优化器参数 `optim_args` 的声明。从顶层来看，在优化过程中，每当看到一个新参数时，带有 `optim_args` 指定参数的构造器 `optim_constructor` 就被用来实例化一个指定类型的新优化器。

很多用户有可能不会直接和 `PyroOptim` 打交道，而是与 `optim/__init__.py` 中定义的别名打交道。让我们看看是怎么回事。由两种途径来声明优化器的参数。

在稍简单的应用中， `optim_args` 是一个为所有参数声明了实例化 PyTorch 优化器所需参数的固定字典：

```python
from pyro.optim import Adam

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)
```
第二种途径允许为更好的控制水平声明参数。此时用户必须指定一个可调用对象，该可调用对象在为新看到的参数创建优化器时，将被 Pyro 调用。此可调用对象必须具有以下签名：

1. `module_name`:  那些包含参数的模块的 Pyro 名称（ 如果有的化）
2. `param_name`:  那些参数的 Pyro 名称

这给用户为不同参数自定义（如学习率）的能力。如果向学习如何利用这种水平的控制，请参见 [基线的讨论](005_svi_part_iii.ipynb) 。这里有一个简单例子来展示该 API：

```python
from pyro.optim import Adam

def per_param_callable(param_name):
    if param_name == 'my_special_parameter':
        return {"lr": 0.010}
    else:
        return {"lr": 0.001}

optimizer = Adam(per_param_callable)
```

上例简单地告诉 Pyro 为参数 `my_special_parameter` 使用 `0.010` 的学习率，为其他参数使用学习率 `0.001` 。

## 7 一个简单案例


我们以一个简单的例子结束。你现在有一个双面硬币，想确定硬币是否公平，即它是否以相同频率出现正面或反面。根据以下两个观察，您对硬币可能的公平性有一个先验的信念：

- 这是美国铸币局发行的标准 0.25 元硬币

- 这枚硬币在使用多年后有点破损

因此，虽然你期望硬币在生产时非常公平，但也允许其公平性偏离完美的 1:1 。因此，如果结果表明硬币以 11:10 的比例偏向于正面，您也不会感到惊讶。相比之下，如果结果表明硬币以 5:1 的比例偏向正面，您会感到非常惊讶。

为了把它变成一个概率模型，我们将正面和反面编码为 `1` 和 `0` 。我们将硬币的公平性编码为实数 $f$，其中 $f$ 满足 $f \in [0.0, 1.0]$ ， $f=0.50$ 对应于完全公平的硬币。

我们对 $f$ 的先验信念将通过贝塔分布进行编码，特别是 $\rm{Beta}(10,10)$，它是区间 $[0.0, 1.0]$ 上的对称概率分布，在 $f=0.5$ 处达到峰值。

```{raw-cell}
:raw_mimetype: text/html

<center><figure><img src="_static/img/beta.png" style="width: 300px;"><figcaption> <font size="-1"><b>Figure 1</b>: The distribution Beta that encodes our prior belief about the fairness of the coin. </font></figcaption></figure></center>
```

为了解比先验更精确的硬币公平性，我们需要做实验并收集一些数据。假设抛硬币 10 次并记录每次抛硬币的结果， 收集在列表 `data` 中，则相应的模型由下式给出：

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

这里有一个隐变量  (`'latent_fairness'` )，它服从分布 $\rm{Beta}(10, 10)$ 。以该随机变量为条件，我们使用伯努利似然来观察每个数据点。请注意，每个观察值在 Pyro 中都被分配了一个唯一的名称。

下一个任务是定义一个相应的引导函数，即隐变量 $f$ 的变分分布。这里唯一真正的要求是 $q(f)$ 应该是 $[0.0, 1.0]$ 范围内的概率分布，因为 $f$ 在该范围之外没有任何意义。一个简单的选择是使用由 $\alpha_q$ 和 $\beta_q$ 参数化的另一个贝塔分布。实际上，在这种特殊情况下，这是一种“正确”的选择，因为伯努利分布和贝塔分布是共轭的，这意味着后验分布也是贝塔分布。在 Pyro 中，我们写：

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

- 我们已经注意随机变量的名称在模型和引导之间完全对齐。

- `model(data)` 和 `guide(data)` 采用相同的参数。 

- 变分参数是张量 `torch.tensor`。梯度计算标志 `requires_grad` 被 `pyro.param` 自动设置为 `True`。

- 使用了 `constraint=constraints.positive` 来确保 `alpha_q` 和 `beta_q` 在优化过程中保持非负。


现在我们可以继续进行随机变分推断：

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

注意，在 `step()` 方法中，我们传入数据，然后将其传递给模型和引导函数。

目前唯一缺少的是数据。那么让我们创建一些数据并将上面的所有代码片段组合成一个完整的脚本：

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
    # register the two variational parameters with Pyro
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

### 案例输出

```
based on the data and our prior belief, the fairness of the coin is 0.532 +- 0.090
```

这个估计将与精确后验均值进行比较，在这种情况下由 $16/30 = 0.5\bar{3}$ 给出。

请注意，硬币公平性的最终估计介于先验（即 0.50 美元）和经验频率建议的公平性（6/10 美元 = 0.60 美元）之间。

+++

## References

[1] `Automated Variational Inference in Probabilistic Programming`,
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
David Wingate, Theo Weber

[2] `Black Box Variational Inference`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Rajesh Ranganath, Sean Gerrish, David M. Blei

[3] `Auto-Encoding Variational Bayes`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Diederik P Kingma, Max Welling
