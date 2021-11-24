---
jupytext:
  formats: ipynb,md:myst
  text_representation: {extension: .md, format_name: myst, format_version: 0.13, jupytext_version: 1.13.1}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# 随机变分推断 （ $\mathrm{III}$ ）:  ELBO的梯度估计器

## 1 设置

我们已经定义了一个 Pyro 模型，其观测值为 ${\bf x}$ 、隐变量为 ${\bf z}$，联合概率形式为 $p_{\theta}({\bf x}, {\bf z}) = p_{\theta}({\bf x}|{\bf z}) p_{\theta}({\bf z})$。我们还定义了 $q_{\phi}({\bf z})$ 形式的 Pyro 引导函数（即变分分布）。这里 ${\theta}$ 和 $\phi$ 分别是模型和引导函数的变分参数，而最重要的是：这些参数都**不是**需要贝叶斯处理的随机变量，而是需要被优化的对象。

我们想通过最大化 ELBO（证据下界）来最大化对数证据 $\log p_{\theta}({\bf x})$：

$$
{\rm ELBO} \equiv \mathbb{E}_{q_{\phi}({\bf z})} \left [ 
\log p_{\theta}({\bf x}, {\bf z}) - \log q_{\phi}({\bf z})
\right]
$$

为此，我们将在参数空间 $\{ \theta, \phi \}$ 中对 ELBO 采取（随机）梯度步骤（有关此方法的早期工作，请参阅参考资料 [1,2]）。所以我们需要能够计算无偏估计：

$$
\nabla_{\theta,\phi} {\rm ELBO} = \nabla_{\theta,\phi}\mathbb{E}_{q_{\phi}({\bf z})} \left [ 
\log p_{\theta}({\bf x}, {\bf z}) - \log q_{\phi}({\bf z})
\right]
$$

对于具有解析表达方式的 ELBO （如共轭情况），给出梯度的表达式是可以的，但大多数情况下，都无法给出梯度的解析表达形式。那么如何为具有普遍性的随机函数 `model()` 和 `guide()` 做到这一点呢？

为了简化符号，让我们先稍微概括一点，思考下对于任意函数 $f({\bf z})$，如何计算其期望梯度。先不考虑 $\theta$ 和 $\phi$ 之间的任何区别，因此是要计算：

$$
\nabla_{\phi}\mathbb{E}_{q_{\phi}({\bf z})} \left [
f_{\phi}({\bf z}) \right]
$$

先从最简单的案例开始。

## 2 简单案例: 可重参数化的随机变量

假设我们能够像下式一样重参数化一些东西：

$$
\mathbb{E}_{q_{\phi}({\bf z})} \left [f_{\phi}({\bf z}) \right]
=\mathbb{E}_{q({\bf \epsilon})} \left [f_{\phi}(g_{\phi}({\bf \epsilon})) \right]
$$

至关重要的是，我们已经将所有 $\phi$ 的依赖项移到了期望内； $q({\bf \epsilon})$ 是不依赖于 $\phi$ 的固定分布。可以对许多分布（例如正态分布）进行这种重参数化；有关讨论，请参见参考文献 [3]。在这种情况下，可以通过期望直接得到梯度：

$$
\nabla_{\phi}\mathbb{E}_{q({\bf \epsilon})} \left [f_{\phi}(g_{\phi}({\bf \epsilon})) \right]=
\mathbb{E}_{q({\bf \epsilon})} \left [\nabla_{\phi}f_{\phi}(g_{\phi}({\bf \epsilon})) \right]
$$

假设 $f(\cdot)$ 和 $g(\cdot)$ 足够平滑，我们就可以通过对这个期望进行蒙特卡罗估计来获得梯度的无偏估计。


## 3 复杂案例：不可重参数化的随机变量

如果我们不能进行上述重参数化怎么办？不幸的是，所有离散型分布在内的大部分分布都无法重参数化。此时我们需要采用更复杂的估计器。

我们从扩展梯度开始：

$$
\nabla_{\phi}\mathbb{E}_{q_{\phi}({\bf z})} \left [
f_{\phi}({\bf z}) \right]= 
\nabla_{\phi} \int d{\bf z} \; q_{\phi}({\bf z}) f_{\phi}({\bf z})
$$

使用链式法则可以改写为：

$$
\int d{\bf z} \; \left \{ (\nabla_{\phi}  q_{\phi}({\bf z})) f_{\phi}({\bf z}) + q_{\phi}({\bf z})(\nabla_{\phi} f_{\phi}({\bf z}))\right \} 
$$

此时会遇到了一个问题：我们知道如何从 $q(\cdot)$ 分布中生成样本，但 $\nabla_{\phi} q_{\phi}({\bf z})$ 甚至都不是一个有效的概率分布。所以需要调整公式，使其成为相对于 $q(\cdot)$ 的期望形式：

$$
 \nabla_{\phi}  q_{\phi}({\bf z}) = 
q_{\phi}({\bf z})\nabla_{\phi} \log q_{\phi}({\bf z})
$$

它允许我们重写梯度为：

$$
\mathbb{E}_{q_{\phi}({\bf z})} \left [
(\nabla_{\phi} \log q_{\phi}({\bf z})) f_{\phi}({\bf z}) + \nabla_{\phi} f_{\phi}({\bf z})\right]
$$

这种形式的梯度估计器被称为强化估计器（ REINFORCE estimator）或评分函数估计器或似然比率估计器，非常适合于简单的蒙特卡洛估计。

注意打包这个结果的一种方式（方便实现）是引入一个代理目标函数（Surrogate Objective Function）。

$$
{\rm surrogate \;objective} \equiv
\log q_{\phi}({\bf z}) \overline{f_{\phi}({\bf z})} + f_{\phi}({\bf z})
$$  

这里的 “ $\bar{}$  ” 符号表示该项可视为常数项（即不用对 $\phi$ 做微分）。为了获得（单样本）蒙特卡罗梯度估计，我们对隐变量进行采样，计算代理目标，并进行微分。其结果是 $\nabla_{\phi}\mathbb{E}_{q_{\phi}({\bf z})} [f_{\phi}({\bf z}) ]$ 。在表达式中：

$$
\nabla_{\phi} {\rm ELBO} = \mathbb{E}_{q_{\phi}({\bf z})} \left [ 
\nabla_{\phi} ({\rm surrogate \; objective}) \right]
$$

## 4 方差问题

我们现在有了一个估计目标函数期望无偏梯度的通用方法。不幸的是，在更一般的情况下，$q(\cdot)$ 包括不可重参数化的随机变量，该估计量往往具有较高的方差。事实上，在许多感兴趣的情况下，方差是大到估计器已经无法使用。因此，需要减少方差的策略（有关讨论，请参见参考文献 [4]）。我们将采取两种策略：

第一种策略利用了代价函数 $f(\cdot)$ 的特定结构。

第二种策略有效地引入了一种减少方差的方法，该方法使用了 $\mathbb{E}_{q_{\phi}({\bf z})} [ f_{\phi}({\bf z} )]$。因此，它有点类似于在随机梯度下降中使用动量。

### 4.1 策略一：通过依赖结构减少差异

在上面的讨论中，我们坚持使用一般性的代价函数 $f_{\phi}({\bf z})$ 。我们可以继续这样，但为了具体起见，让我们放大。在随机变分推断的情况下，我们对特定形式的代价函数感兴趣：

$$
\log p_{\theta}({\bf x} | {\rm Pa}_p ({\bf x})) +
\sum_i \log p_{\theta}({\bf z}_i | {\rm Pa}_p ({\bf z}_i)) - \sum_i \log q_{\phi}({\bf z}_i | {\rm Pa}_q ({\bf z}_i))
$$

其中已经把对数比率 $\log p_{\theta}({\bf x}, {\bf z})/q_{\phi}({\bf z})$ 分解成了一个观测的对数似然片段和不同隐变量 $\{{\bf z}_i \}$ 的和。我们引入了符号 ${\rm Pa}_p (\cdot)$ 和 ${\rm Pa}_q (\cdot)$ 分别表示模型和引导函数中给定随机变量的父项. （读者可能会担心在一般性的随机函数中，依赖的合适概念到底是什么；这里我们仅指单个执行跟踪中的常规 ol' 依赖）。关键是代价函数中的不同项对随机变量 $\{ {\bf z}_i \}$ 有不同的依赖性，这是我们可以利用的。

长话短说，对于任何不可重参数化的隐变量 ${\bf z}_i$ ，代理目标将有一个项：

$$
\log q_{\phi}({\bf z}_i) \overline{f_{\phi}({\bf z})} 
$$

事实证明，我们可以去掉 $\overline{f_{\phi}({\bf z})}$ 中的一些项，并且仍然能够​​得到一个无偏梯度估计量；此外，这样做通常会减少方差。特别是（详见参考文献 [4]），我们可以删除 $\overline{f_{\phi}({\bf z})}$ 中任意一项，只要该项不在隐变量 ${\bf z}_i$ 的下游（指相对于引导函数中依赖结构的下游）。请注意，这种“通过分析处理某些随机变量以减少方差”的通用技巧，通常以 `Rao-Blackwellization` 的名称出现。

在 Pyro 中，所有这些逻辑都由`SVI` 类自动处理。特别是，只要我们使用 `TraceGraph_ELBO` 损失，Pyro 将跟踪模型执行轨迹中的依赖结构，并指导和构建一个代理目标，该目标已经删除了所有不必要的项：

```python
svi = SVI(model, guide, optimizer, TraceGraph_ELBO())
```
请注意，利用此依赖信息需要额外的计算，因此 `TraceGraph_ELBO` 应当仅用于您的模型具有无法重参数化的变量的情况；在大多数应用中，`Trace_ELBO` 就足够了。

+++

#### 案例：一个 Rao-Blackwellization 的例子

假设我们有一个包含 $K$ 个分量的高斯混合模型。对于每个数据点： (i) 首先对分量分布 $k \in [1,...,K]$ 进行采样； (ii) 使用 $k^{\rm th}$ 分量的分布得到观测数据点。此模型的最简单编程方法如下：

```python
ks = pyro.sample("k", dist.Categorical(probs)
                          .to_event(1))
pyro.sample("obs", dist.Normal(locs[ks], scale)
                       .to_event(1),
            obs=data)
```

由于用户没有在模型中标记任何条件独立性， Pyro 的 `SVI` 类构建的梯度估计器就无法利用 `Rao-Blackwellization`，其结果是梯度估计器将倾向于受到高方差的影响。为解决该问题，用户需要明确地标记条件独立性。令人高兴的是，这并不会带来太大工作量：

```python
# mark conditional independence 
# (assumed to be along the rightmost tensor dimension)
with pyro.plate("foo", data.size(-1)):
    ks = pyro.sample("k", dist.Categorical(probs))
    pyro.sample("obs", dist.Normal(locs[ks], scale),
                obs=data)
```      

+++

#### 补充：Pyro 中的依赖跟踪

最后，谈一谈依赖跟踪。在包含任意 Python 代码的随机函数中跟踪依赖性有点棘手。目前在 Pyro 中实现的方法类似于 `WebPPL` 中使用的方法（参见参考文献 [5]）。

简而言之，采用了一个比较保守的、依靠顺序排序的依赖概念，如果随机变量 ${\bf z}_2$ 在给定随机函数中跟随 ${\bf z}_1$ ，那么 ${\bf z}_2$ 有可能依赖于 ${\bf z}_1$ ，因此被假定为依赖。为了减轻这种依赖跟踪可能得出的过于粗略的结论，Pyro 提供了将事物声明为独立的构造元语，即 `plate` 和 `markov`（[参见上一教程](svi_part_ii.ipynb)）。

对于具有不可重参数化变量的用例，重要的是利用这些构造元语来充分发挥 `SVI` 类中方差减少手段的优势。在某些情况下，考虑对随机函数中的随机变量重新排序可能也是值得的（如果可能）。

### 4.2 策略二：使用依赖于数据的基线减少方差

Pyro 中的 ELBO 梯度估计器中，第二种减少方差的策略是采用基线（参见例如参考文献 [6]）。 它实际上使用了与上面讨论的方差减少策略相同的数学基础，不过现在我们将添加项而不是删除项。 不是删除倾向于增加方差的零期望项，而是添加专门选择的、用于减少方差的零期望项。 因此，这是一种控制变量（Control Variate）策略。

更详细地说，这个想法是利用这样一个事实，即对于任何常数 $b$，以下恒等式成立：

$$
\mathbb{E}_{q_{\phi}({\bf z})} \left [\nabla_{\phi}
(\log q_{\phi}({\bf z}) \times b) \right]=0
$$

这是因为 $q(\cdot)$ 被标准化了：

$$
\mathbb{E}_{q_{\phi}({\bf z})} \left [\nabla_{\phi}
\log q_{\phi}({\bf z}) \right]=
 \int \!d{\bf z} \; q_{\phi}({\bf z}) \nabla_{\phi}
\log q_{\phi}({\bf z})=
 \int \! d{\bf z} \; \nabla_{\phi} q_{\phi}({\bf z})=
\nabla_{\phi} \int \! d{\bf z} \;  q_{\phi}({\bf z})=\nabla_{\phi} 1 = 0
$$

这意味着我们可以将代理目标中的任何项

$$
\log q_{\phi}({\bf z}_i) \overline{f_{\phi}({\bf z})} 
$$

替换为：

$$\log q_{\phi}({\bf z}_i) \left(\overline{f_{\phi}({\bf z})}-b\right) $$

这样做不会影响梯度估计器的均值，但会影响方差。如果我们明智地选择 $b$，可以希望减少方差。事实上，$b$ 不需要是一个常数：它可以依赖于 ${\bf z}_i$ 上游的任何随机选择。

#### Pyro 中的基线

用户可以通过多种方式指示 Pyro 在随机变分推理上下文中使用基线。由于基线可以附加到任何不可重参数化的随机变量，因此当前基线接口处于 `pyro.sample` 语句的级别。特别是基线接口使用了一个名为 `baseline` 的接口，它是一个指定基线选项的字典。请注意，仅在引导函数中（而不是在模型中）为示例语句指定基线才有意义。

##### （1）衰减平均基线（Decaying Average Baseline）

The simplest baseline is constructed from a running average of recent samples of $\overline{f_{\phi}({\bf z})}$. In Pyro this kind of baseline can be invoked as follows

最简单的基线是根据 $\overline{f_{\phi}({\bf z})}$ 的最近运行样本的平均值构建。在 Pyro 中，可以如下调用这种基线：

```python
z = pyro.sample("z", dist.Bernoulli(...), 
                infer=dict(baseline={'use_decaying_avg_baseline': True,
                                     'baseline_beta': 0.95}))
```

可选参数 `baseline_beta` 指定衰减平均值的衰减率（默认值：`0.90`）。

##### （2）神经基线（Neural Baselines）

在某些情况下，衰减平均基线效果很好。在另外一些情况下，使用依赖于上游随机性的基线对于获得良好的方差减少更加重要。构建此类基线的一种强大方法是使用可以在学习过程中进行调整的神经网络。 

Pyro 提供了两种方法来指定这样的基线（有关扩展示例，请参阅 [AIR 教程](air.ipynb)）。

首先，用户需要决定基线将使用哪些输入（例如，正在考虑的当前数据点或先前采样的随机变量）。然后用户需要构造一个封装了基线计算的 `nn.Module` 。可能看起来像这样：

```python
class BaselineNN(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super().__init__()
        self.linear = nn.Linear(dim_input, dim_hidden)
        # ... finish initialization ...

    def forward(self, x):
        hidden = self.linear(x)
        # ... do more computations ...
        return baseline
```

然后，假设 `BaselineNN` 对象 `baseline_module` 已在其他地方初始化，在引导函数中我们将有如下的内容：

```python
def guide(x):  # here x is the current mini-batch of data
    pyro.module("my_baseline", baseline_module)
    # ... other computations ...
    z = pyro.sample("z", dist.Bernoulli(...), 
                    infer=dict(baseline={'nn_baseline': baseline_module,
                                         'nn_baseline_input': x}))
```

参数 `nn_baseline` 告诉 Pyro 使用哪个 `nn.Module` 来构建基线。在后端，参数 `nn_baseline_input` 被送入模块的前向（forward）方法中用于计算基线 $b$。请注意，基线模块需要调用 `pyro.module` 在 Pyro 中注册，以便 Pyro 知道模块内的可训练参数。

在 Pyro 中构建了一个如下形式的损失：

$$
{\rm baseline\; loss} \equiv\left(\overline{f_{\phi}({\bf z})} - b  \right)^2
$$

该损失用于调整神经网络的参数。没有定理表明在这是最佳损失函数，但在实践中效果很好。就像衰减平均基线一样，我们的想法是跟踪平均值 $\overline{f_{\phi}({\bf z})}$ 的基线将有助于减少方差。`SVI` 在基线损失上迈出一步，同时在 ELBO 上迈出一步。

请注意，在实践中，对基线参数使用不同的超参数（例如更高的学习率）可能很重要。在 Pyro 中，可以按如下方式完成：

```python
def per_param_args(param_name):
    if 'baseline' in param_name:
        return {"lr": 0.010}
    else:
        return {"lr": 0.001}
    
optimizer = optim.Adam(per_param_args)
```

请注意，为了使整个过程正确，基线参数应仅通过基线损失进行优化。同样，模型和引导函数的参数只能通过 ELBO 进行优化。为了确保这种情况，`SVI` 在底层会从 autograd 图中分离了进入 ELBO 的基线 $b$。此外，由于神经基线的输入可能取决于模型和引导函数的参数，因此在将输入馈入神经网络之前，它们也会从 autograd 图中被分离出来。

最后，还有一种可供用户指定神经基线的替代方法。只需使用参数`baseline_value`：

```python
b = # do baseline computation
z = pyro.sample("z", dist.Bernoulli(...), 
                infer=dict(baseline={'baseline_value': b}))
```

This works as above, except in this case it's the user's responsibility to make sure that any autograd tape connecting $b$ to the parameters of the model and guide has been cut. Or to say the same thing in language more familiar to PyTorch users, any inputs to $b$ that depend on $\theta$ or $\phi$ need to be detached from the autograd graph with `detach()` statements.

这和上面一样地工作，除了在这种情况下，用户有责任确保将 $b$ 到模型和引导函数参数的任何 autograd 连接已被切断。或者用 PyTorch 用户更熟悉的语言来描述，任何依赖于 $\theta$ 或 $\phi$ 的 $b$ 输入都需要使用 `detach()` 语句从 autograd 图中分离。


##### （3）使用基线的完整案例

回想一下，在[第一个 SVI 教程](svi_part_i.ipynb) 中，我们考虑了硬币翻转的`伯努利-贝塔`模型。因为贝塔随机变量是不容易重参数化，所以相应的 ELBO 梯度可能噪声较大。为此当时使用了一个能够提供（近似）重参数化梯度的贝塔分布来解决问题。下面将展示了一个简单的衰减平均基线案例，能够在贝塔分布无法重参数化的情况下减少方差（ 采用评分拟函数 ELBO 梯度估计器）。在此过程中，我们还使用了向量化 `plate` 的方式编写模型。

我们将不直接比较梯度的方差，而是查看 SVI 收敛需要多少步。回想一下，对于这个特定模型（因为共轭），我们可以计算精确的后验。因此，为了评估基线在这种情况下的效用，我们设置了以下简单的实验：

我们以一组指定的变分参数初始化引导函数。然后进行 SVI，直到变分参数达到精确后验参数的指定容差范围。我们在使用和不使用衰减平均基线的情况下都这样做，然后比较两种情况下所需的梯度步数。

下面是完整代码：

```{code-cell} ipython3
import os
import torch
import torch.distributions.constraints as constraints
import pyro
import pyro.distributions as dist
# Pyro also has a reparameterized Beta distribution so we import
# the non-reparameterized version to make our point
from pyro.distributions.testing.fakes import NonreparameterizedBeta
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
import sys

assert pyro.__version__.startswith('1.7.0')

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
max_steps = 2 if smoke_test else 10000


def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).item()


class BernoulliBetaExample:
    def __init__(self, max_steps):
        # the maximum number of inference steps we do
        self.max_steps = max_steps
        # the two hyperparameters for the beta prior
        self.alpha0 = 10.0
        self.beta0 = 10.0
        # the dataset consists of six 1s and four 0s
        self.data = torch.zeros(10)
        self.data[0:6] = torch.ones(6)
        self.n_data = self.data.size(0)
        # compute the alpha parameter of the exact beta posterior
        self.alpha_n = self.data.sum() + self.alpha0
        # compute the beta parameter of the exact beta posterior
        self.beta_n = - self.data.sum() + torch.tensor(self.beta0 + self.n_data)
        # initial values of the two variational parameters
        self.alpha_q_0 = 15.0
        self.beta_q_0 = 15.0

    def model(self, use_decaying_avg_baseline):
        # sample `latent_fairness` from the beta prior
        f = pyro.sample("latent_fairness", dist.Beta(self.alpha0, self.beta0))
        # use plate to indicate that the observations are
        # conditionally independent given f and get vectorization
        with pyro.plate("data_plate"):
            # observe all ten datapoints using the bernoulli likelihood
            pyro.sample("obs", dist.Bernoulli(f), obs=self.data)

    def guide(self, use_decaying_avg_baseline):
        # register the two variational parameters with pyro
        alpha_q = pyro.param("alpha_q", torch.tensor(self.alpha_q_0),
                             constraint=constraints.positive)
        beta_q = pyro.param("beta_q", torch.tensor(self.beta_q_0),
                            constraint=constraints.positive)
        # sample f from the beta variational distribution
        baseline_dict = {'use_decaying_avg_baseline': use_decaying_avg_baseline,
                         'baseline_beta': 0.90}
        # note that the baseline_dict specifies whether we're using
        # decaying average baselines or not
        pyro.sample("latent_fairness", NonreparameterizedBeta(alpha_q, beta_q),
                    infer=dict(baseline=baseline_dict))

    def do_inference(self, use_decaying_avg_baseline, tolerance=0.80):
        # clear the param store in case we're in a REPL
        pyro.clear_param_store()
        # setup the optimizer and the inference algorithm
        optimizer = optim.Adam({"lr": .0005, "betas": (0.93, 0.999)})
        svi = SVI(self.model, self.guide, optimizer, loss=TraceGraph_ELBO())
        print("Doing inference with use_decaying_avg_baseline=%s" % use_decaying_avg_baseline)

        # do up to this many steps of inference
        for k in range(self.max_steps):
            svi.step(use_decaying_avg_baseline)
            if k % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            # compute the distance to the parameters of the true posterior
            alpha_error = param_abs_error("alpha_q", self.alpha_n)
            beta_error = param_abs_error("beta_q", self.beta_n)

            # stop inference early if we're close to the true posterior
            if alpha_error < tolerance and beta_error < tolerance:
                break

        print("\nDid %d steps of inference." % k)
        print(("Final absolute errors for the two variational parameters " +
               "were %.4f & %.4f") % (alpha_error, beta_error))

# do the experiment
bbe = BernoulliBetaExample(max_steps=max_steps)
bbe.do_inference(use_decaying_avg_baseline=True)
bbe.do_inference(use_decaying_avg_baseline=False)
```

**Sample output:**
```
Doing inference with use_decaying_avg_baseline=True
....................
Did 1932 steps of inference.
Final absolute errors for the two variational parameters were 0.7997 & 0.0800
Doing inference with use_decaying_avg_baseline=False
..................................................
Did 4908 steps of inference.
Final absolute errors for the two variational parameters were 0.7991 & 0.2532
```

+++

For this particular run we can see that baselines roughly halved the number of steps of SVI we needed to do. The results are stochastic and will vary from run to run, but this is an encouraging result. This is a pretty contrived example, but for certain model and guide pairs, baselines can provide a substantial win. 

+++

## 参考文献

[1] `Automated Variational Inference in Probabilistic Programming`,
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
David Wingate, Theo Weber

[2] `Black Box Variational Inference`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Rajesh Ranganath, Sean Gerrish, David M. Blei

[3] `Auto-Encoding Variational Bayes`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Diederik P Kingma, Max Welling

[4] `Gradient Estimation Using Stochastic Computation Graphs`,
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
    John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel
    
[5] `Deep Amortized Inference for Probabilistic Programs`
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Daniel Ritchie, Paul Horsfall, Noah D. Goodman

[6] `Neural Variational Inference and Learning in Belief Networks`
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Andriy Mnih, Karol Gregor