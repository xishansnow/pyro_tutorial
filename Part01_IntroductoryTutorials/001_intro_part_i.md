---
jupytext:
  formats: ipynb,md:myst
  text_representation: {extension: .md, format_name: myst, format_version: 0.13, jupytext_version: 1.13.1}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

#  Pyro 中的概率模型

## 1 概述
概率程序的基本单位是*随机函数*。这是一个任意的 Python 可调用对象，它结合了两种成分：

- 确定性 Python 代码；

- 调用随机数生成器的元随机函数


具体来说，随机函数可以是具有 `__call__()` 方法的任何 Python 对象，例如：函数、方法或 PyTorch 的 `nn.Module`。

在整个教程和文档中，我们经常会称随机函数为`模型`，因为随机函数可用来表示生成数据的过程。将模型表示为随机函数意味着：**模型可以像常规 Python 可调用对象一样组合、重用、导入和序列化模型**。

```{code-cell}
import torch
import pyro

pyro.set_rng_seed(101)
```

## 2 PyTorch 中的元随机函数

元随机函数（或分布）是一类重要的随机函数，我们可以使用它显式地计算出指定输入的概率输出。从 PyTorch 0.4 和 Pyro 0.2 开始，Pyro 使用 [PyTorch 的概率分布库](http://pytorch.org/docs/master/distributions.html)。您还可以使用 [PyTorch 的 `torch.distributions.transforms` 软件包](http://pytorch.org/docs/master/distributions.html#module-torch.distributions.transforms) 创建自定义的分布。

在 PyTorch 中使用元随机函数很容易。例如，要从单位正态分布 $\mathcal{N}(0,1)$ 中抽取样本 `x`，我们可以执行以下操作：

```{code-cell}
loc = 0.   # mean zero
scale = 1. # unit variance
normal = torch.distributions.Normal(loc, scale) # create a normal distribution object
x = normal.rsample() # draw a sample from N(0,1)
print("sample", x)
print("log prob", normal.log_prob(x)) # score the sample from N(0,1)
```

这里，`torch.distributions.Normal` 是 `Distribution` 类的一个实例，它接受概率分布的参数并提供样本和评分方法。 

Pyro 的概率分布软件库 `pyro.distributions` 其实是对 `torch.distributions` 的一个瘦包装，因为我们想在推断过程中利用 PyTorch 的快速张量计算和自动梯度计算（ `autograd` ）功能。

+++

## 3 从简单模型开始

**首先应当清楚， Pyro 的所有概率程序都是通过组合元随机函数和确定性计算来构建的**。我们之所以对概率编程感兴趣，正是希望能够利用它来对现实世界中的事物建模，因此让我们从一个具体事务的模型开始。

假设现在有一些包含日均气温和云量的数据，我们想推断温度与晴/阴天之间的相互作用，那么描述该数据某个可能生成过程的随机函数可以简单由下面 PyTorch 相关的代码给出：

```{code-cell}
def weather():
    cloudy = torch.distributions.Bernoulli(0.3).sample()
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()
    return cloudy, temp.item()
```

让我们逐行浏览一遍。首先，在第 2 行中定义了一个二值随机变量 `cloudy`，它由参数为 $0.3$ 的伯努利分布抽样得出。由于伯努利分布返回 `0` 或 `1`，所以在第 3 行我们将其值转换成了字符串，以便更容易解析 `weather` 的返回值。根据这个模型，30% 的时间是阴天，70% 的时间是晴天。在第 4-5 行中定义了一些即将用于第 6 行中温度采样的参数。这些参数由第 2 行中采样的`cloudy` 值确定。例如，阴天平均温度为 $55$ 度（华氏度）、晴天平均温度为 $75$ 度。最后，在第 7 行返回了天气 `cloudy` 和气温 `temp` 两个随机变量。

目前的 `weather` 随机函数主要用于生成模拟数据，仅调用了 PyTorch，而完全和 Pyro 无关。但如果想将此模型用于生成模拟数据以外的任何其他任务（如：推断、预测性分布等），则必须将其转换为 Pyro 程序。

+++

## 4  Pyro 的 `sample` 元语

为了将 `weather` 随机函数变成 Pyro 程序，需要将 `torch.distribution` 替换为 `pyro.distribution` ，并将对 `.sample()` 和 `.rsample()` 的调用替换为对 `pyro.sample()` 的调用。

`pyro.sample` 是 Pyro 中的核心元语之一，使用 `pyro.sample` 就像 PyTorch 中调用元随机函数一样简单，但有一个重要区别：

```{code-cell}
x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale))
print(x)
```

代码中调用的 `Pyro.sample()` 就像直接调用 `torch.distributions.Normal().rsample()` 一样，会从正态分布中返回一个样本。两者之间的关键区别在于：**Pyro 中的样本是命名样本**。 Pyro 的后端会使用样本名称来唯一地标识样本语句，并且在运行时，根据封闭形式随机函数的使用方式改变语句的行为。正如将看到的，这是 Pyro 实现各种推断算法的基础。

+++

现在我们已经引入了 `pyro.sample` 和 `pyro.distributions`，可以将上述简单模型重写为 Pyro 程序：

```{code-cell}
def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

for _ in range(3):
    print(weather())
```

从程序上讲，`weather()`  还是一个非确定性的 Python 可调用对象，因为每次调用它都会返回不同的随机样本。但是由于使用 `pyro.sample` 激发了随机性，所以它能做的远不止于此。事实上，Pyro 版本的 `weather()` 指定了两个命名随机变量 `cloudy` 和 `temp` 的联合概率分布。因此，该随机函数定义了一个概率模型，而我们可以使用概率知识对其进行推断。例如，我们可能会问：如果观察到 $70$ 度的温度，那么阴天的可能性有多大？如何制定和回答此类问题将是下一个[推断教程](003_svi_part_i.ipynb) 的主题。

+++

## 4 通用性: 随机递归、高阶随机函数和随机控制流

我们现在已经看到了如何定义一个简单的模型。构建它很容易。例如：

```{code-cell}
def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream
```

这种任何程序员都很熟悉的模块化能力，显然非常强大。但是它是否足够强大以涵盖我们想要表达的所有类型模型？事实证明，因为 Pyro 嵌入在 Python 中，随机函数可以包含任意复杂的确定性 Python，并且随机性可以自由地影响控制流。

例如，可以构造一个非确定性终止递归过程的递归函数，前提是在调用 `pyro.sample` 时，注意传递唯一的样本名称。例如，我们可以定义一个几何分布来为失败次数计数，直到第一次成功：

```{code-cell}
def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)
    
print(geometric(0.5))
```

请注意，`geometric()` 中的 `x_0`、`x_1` 等名称是动态生成的，不同的执行可以拥有不同数量的命名随机变量。

我们也可以自由定义**将随机函数作为输入或输出**的随机函数：

```{code-cell}
def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y

def make_normal_normal():
    mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn     # 将随机函数作为输出

print(make_normal_normal()(1.))
```

这里的 `make_normal_normal()` 是一个随机函数，它接受一个参数，并在执行时生成三个命名的随机变量。

事实上，Pyro 还支持迭代、递归、高阶函数等任意 Python 代码，并且能够与随机控制流结合，这意味着 Pyro 随机函数是通用的，即它们可用于表示任何可计算的概率分布，这一点非常强大。

值得强调的是，这也是 Pyro 选择构建在 PyTorch 之上的一个原因：动态计算图是通用模型的重要组成部分，这些模型可以从 GPU 加速的张量计算中受益。

+++

## 下一步

我们展示了如何使用随机函数和原子分布来表示 Pyro 中的模型。为了从数据中学习模型并对其进行推断，我们需要推理功能。而这是[下一个教程](002_intro_part_ii.ipynb) 的主题。
