---
jupytext:
  formats: ipynb,md:myst
  text_representation: {extension: .md, format_name: myst, format_version: 0.13, jupytext_version: 1.13.1}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# 随机变分推断 （ $\mathrm{II}$ ）: 条件独立、二次抽样和摊销

## 1 目标： 将 SVI 扩展到大型数据集

对于具有 $N$ 个观测值的模型，运行 `model` 和 `guide` 并构建 ELBO 涉及评估对数概率密度 $\log p(\cdot)$，其复杂性随着 $N$ 增加严重加大。如果想扩展到大型数据集，这是必须解决的一个问题。幸运的是，ELBO 目标函数天然支持二次采样，前提是模型/引导函数具有一些可以利用的条件独立结构。例如，在给定隐变量的情况下，观测是条件独立的。ELBO 中的对数似然项可以近似为：

$$ \sum_{i=1}^N \log p({\bf x}_i | {\bf z}) \approx  \frac{N}{M}
\sum_{i\in{\mathcal{I}_M}} \log p({\bf x}_i | {\bf z})  $$

其中 $\mathcal{I}_M$ 为大小为 $M$ 的一个小批次，且 $M<N$ （讨论见参考文献【1，2】）。那么在 Pyro 该怎么做呢?

## 2 在 Pyro 中标记条件独立性

如果用户想在 Pyro 中实现上述能力，首先需要确保模型和引导的写法符合 Pyro 使用条件独立性的形式。 Pyro 提供了两种用于标记条件独立性的语言元语：`plate` 和 `markov`。让我们从两者中较简单的开始。

### 2.1 顺序 `plate`

先回到[之前使用过得案例](003_svi_part_i.ipynb)。 方便起见，将 `model` 的主要逻辑复现如下：

```python
def model(data):
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data using pyro.sample with the obs keyword argument
    for i in range(len(data)):
        # observe datapoint i using the bernoulli likelihood
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])
```        

这个模型的观测数据条件独立于给定的隐变量 `latent_fairness` 。为了在 Pyro 中显式标记这种独立性 ，最基本的作法是将 Python 中内置的 `range` 替换为 Pyro 的 `plate` 构造:

```python
def model(data):
    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data [WE ONLY CHANGE THE NEXT LINE]
    for i in pyro.plate("data_loop", len(data)):  
        # observe datapoint i using the bernoulli likelihood
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])
```

可以看见， `pyro.plate` 和 `range` 非常相似，除了一点主要的区别：每次对 `plate` 的调用都需要用户先提供一个具有唯一性的名称，第二个参数才和 `range` 一样是个整数。

到现在为止还挺好。 Pyro 现在可以利用给定隐变量时观测数据的条件独立性了。但其实际上如何工作呢？`pyro.plate` 基本上是使用上下文管理器实现的。在每次执行 `for` 循环体时，都会进入一个新的（有条件的）独立上下文，然后在 `for` 循环体结束时退出。让我们非常明确说明以下几点：

- 每次观测的 `pyro.sample` 语句发生在 `for` 循环体的不同执行上下文中，因此，Pyro 将每次观测标记为独立的

- 该独立性是在给定 `latent_fairness` 时的条件独立性，因为 `latent_fairness` 是在`data_loop` 上下文之外采样的。

在继续前，让我们提醒下，在使用顺序 `plate` 时要避免的一些问题。考虑上述代码片段的以下变体：

```python
# WARNING do not do this!
my_reified_list = list(pyro.plate("data_loop", len(data)))
for i in my_reified_list:  
    pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])
```

这不会获得预期想要的结果，因为 `list()` 将在调用 `pyro.sample` 语句之前，就进入和退出`data_loop` 的上下文。同样，用户需要注意不要跨上下文管理器的边界调用随机变量进行计算，这可能导致错误。例如，`pyro.plate` 不适用于循环的每次迭代依赖于前一次迭代的时间序列模型；在这种情况下，应该使用 `range` 或 `pyro.markov` 来代替。

### 2.2 向量化 `plate`

概念上讲，向量化的 `plate` 与顺序 `plate` 是相同的，只是它能够执行向量化操作（ 就像 `torch.arange` 和 `range` ）。因此，与出现在顺序 `plate` 中的显式 `for` 循环相比，它会实现大幅加速。让我们看看对于运行示例来说其表现如何。首先，需要 `data` 采用张量的形式：

```python
data = torch.zeros(10)
data[0:6] = torch.ones(6)  # 6 heads and 4 tails
```

然后，我们有：

```python
with pyro.plate('observe_data'):
    pyro.sample('obs', dist.Bernoulli(f), obs=data)
```

让我们比较和逐点的顺序 `plate` 方法做一个对比：

- 两种模式都要求用户指定一个唯一的名称。

- 此代码片段仅引入了一个（观测到的）随机变量（ 即`obs` ），因为会同时考虑整个张量。 

- 在向量化情况下不需要迭代器，所以无需指定 `plate` 上下文中涉及的张量长度

请注意，在顺序 `plate` 中提到的各种问题也适用于向量化 `plate`。

+++

## 3 二次采样

现在知道如何在 Pyro 中标记条件独立性了，这本身很有用。请参阅[随机变分推断第 III 部分](005_svi_part_iii.ipynb) 中的依赖项跟踪部分。但我们有时候也希望进行二次采样，以便可以在大型数据集上进行随机变分推断。根据模型和引导的结构不同，Pyro 支持多种二次采样的方法。

### 3.1  `plate` 的自动二次采样

先看看最简单的情况，在此情况下，我们可以对 `plate` 使用一两个额外的参数进行二次采样：

```python
for i in pyro.plate("data_loop", len(data), subsample_size=5):
    pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])
```    

这里我们仅使用了参数 `subsample_size` 就可以实现二次采样。每当运行 `model()` 时，只评估 `data` 中随机选择的 5 个数据点的对数似然；此外，对数似然值将自动按适当的因子 $\tfrac{10}{5} = 2$ 进行缩放。 向量化的 `plate` 又如何呢？方法完全类似：

```python
with pyro.plate('observe_data', size=10, subsample_size=5) as ind:
    pyro.sample('obs', dist.Bernoulli(f), 
                obs=data.index_select(0, ind))
```
重要的是，`plate` 现在返回一个索引为 `ind` 的张量，其长度为 5 。注意，除了参数 `subsample_size` 之外，还传递了向量的长度参数 `size`，因此 `plate` 需要知道张量 `data` 的完整大小，以便计算正确的缩放因子。就像顺序 `plate` 一样，用户负责使用 `plate` 提供的索引来选择正确的数据点。

最后，请注意，如果 `data` 在 GPU 上，用户必须将 `device` 参数传递给 `plate`。

### 3.2 自定义 `plate` 的二次采样策略

每次运行上述 `model()` 时，`plate` 都会采样新的二次采样索引。由于这种二次采样是无状态的，因此可能会导致一些问题：对于足够大的数据集，即使经过大量迭代，某些数据点永远不会被选中的可能性也是存在的。为了避免这种情况，用户可以通过使用 `plate` 的 `subsample` 参数来控制二次采样。有关详细信息，请参阅 [文档](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate)。

### 3.3 只有局部随机变量时的二次采样 

我们有一个模型，其联合概率密度由下式定义：

$$
p({\bf x}, {\bf z}) = \prod_{i=1}^N p({\bf x}_i | {\bf z}_i) p({\bf z}_i)  
$$

对于具有这种依赖结构的模型，二次采样引入的比例因子将对 ELBO 中的所有项按相同比例进行缩放。例如，对于传统变分自编码器（ VAE ）就是这种情况。这解释了为什么变分自编码器允许用户完全控制二次采样，并将小批量直接传递给模型和引导； `plate` 仍在使用，但 `subsample_size` 和 `subsample` 不再使用。要查看详细信息，请参阅 [VAE 教程](001_vae.ipynb)。

### 3.4 同时拥有全局和局部随机变量的二次采样

在上面的硬币翻转示例中，`plate` 出现在模型中但没有出现在引导中，因为唯一被二次采样的是观测数据。让我们看一个更复杂的示例，其中二次采样同时出现在模型和引导中。为简单起见，我们保持了一些抽象性，以避免编写完整的模型和引导。

考虑由以下联合分布指定的模型：

$$
p({\bf x}, {\bf z}, \beta) = p(\beta) 
\prod_{i=1}^N p({\bf x}_i | {\bf z}_i) p({\bf z}_i | \beta)  
$$

有 $N$ 条观测数据 $\{ {\bf x}_i \}$ 和 $N$ 个局部隐变量  
$\{ {\bf z}_i \}$ ，同时还有一个全局变量 $\beta$ 。 我们的引导函数被因子化为：

$$
q({\bf z}, \beta) = q(\beta) \prod_{i=1}^N q({\bf z}_i | \beta, \lambda_i)  
$$

这里我们已经显式地引入了 $N$ 个局部变分参数 $\{\lambda_i \}$ ，而其他变分参数为隐式的。模型和引导函数均包含条件独立性，特别是在模型中，给定 $\{ {\bf z}_i \}$ 时观测值 $\{ {\bf x}_i \}$ 是独立的 。 此外，给定 $\beta$ 时，隐变量  $\{\bf {z}_i \}$ 也是独立的。在引导函数方面，给定变分参数 $\{\lambda_i \}$ 和 $\beta$ 隐变量 $\{\bf {z}_i \}$ 是独立的。为了在 Pyro 中标记这些条件独立性并且利用 `plate` 在模型和引导函数中都能够做二次采样。让我们给出使用顺序 `plate` 的基本逻辑线（更完整的代码片段应包含 `pyro.param` 语句等）。 

首先，模型为：

```python
def model(data):
    beta = pyro.sample("beta", ...) # sample the global RV
    for i in pyro.plate("locals", len(data)):
        z_i = pyro.sample("z_{}".format(i), ...)
        # compute the parameter used to define the observation 
        # likelihood using the local random variable
        theta_i = compute_something(z_i) 
        pyro.sample("obs_{}".format(i), dist.MyDist(theta_i), obs=data[i])
```

注意，相对于我们正在运行的抛硬币案例，此处我们在 `plate` 循环的内外部都有 `pyro.sample` 语句。引导函数为：

```python
def guide(data):
    beta = pyro.sample("beta", ...) # sample the global RV
    for i in pyro.plate("locals", len(data), subsample_size=5):
        # sample the local RVs
        pyro.sample("z_{}".format(i), ..., lambda_i)
```
请注意，至关重要的是，引导函数中的索引只会做一次二次抽样； Pyro 后端确保在模型执行期间使用相同的索引集。出于该原因，`subsample_size` 只需要在引导函数中指定即可。

+++

## 4 摊销（Amortization）

让我们再次考虑具有全局变量、局部变量和局部变分参数的模型：

$$
p({\bf x}, {\bf z}, \beta) = p(\beta) 
\prod_{i=1}^N p({\bf x}_i | {\bf z}_i) p({\bf z}_i | \beta)  \qquad \qquad\\
q({\bf z}, \beta) = q(\beta) \prod_{i=1}^N q({\bf z}_i | \beta, \lambda_i)  
$$

对于中小规模的 $N$ ，使用像这样的局部变分参数可能是一个好方法。然而，如果 $N$ 很大，那么优化空间随着 $N$ 而增长的事实可能会变成一个真正的问题。避免随着数据集增长而增长的一种方法是**摊销**。

在摊销方案中，我们不再引入局部变分参数，而是学习一个参数化的函数 $f(\cdot)$ ，并使用具有以下形式的变分分布。

$$
q(\beta) \prod_{n=1}^N q({\bf z}_i | f({\bf x}_i))
$$

函数 $f(\cdot)$ 将给定的观测值映射到一组针对该数据点定制的变分参数上。这要求函数必须足够丰富才能准确捕获到后验，但好处是可以处理大型数据集而无需引入大量的变分参数。

摊销方法还有其他的好处：如在学习期间， $f(\cdot)$ 允许我们有效地在不同数据点之间共享统计能力。请注意，而这正是 [变分自编码器 VAE](vae.ipynb) 中使用的方法。

>注意： 张量形状与向量化 `plate`

本文中对 `pyro.plate` 的使用仅局限于相当简单的场景。例如，不存在 `plate` 嵌套的情况。 当面对更复杂的 `plate` 应用场景时，用户必须非常小心地使用 Pyro 中张量形状的语义。 有关讨论见 [张量形状教程](tensor_shapes.ipynb).

+++

## 参考文献

[1] `Stochastic Variational Inference`,
<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Matthew D. Hoffman, David M. Blei, Chong Wang, John Paisley

[2] `Auto-Encoding Variational Bayes`,<br/>&nbsp;&nbsp;&nbsp;&nbsp;
Diederik P Kingma, Max Welling
