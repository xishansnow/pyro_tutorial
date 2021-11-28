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

#  Pyro 中的张量形状

本教程介绍了 Pyro 对张量维度的组织方法。在开始之前，你应该先熟悉一下[PyTorch 广播机制](http://pytorch.org/docs/master/notes/broadcasting.html)。 在本教程之后，你可能还想阅读一下[枚举类型](http://pyro.ai/examples/enumeration.html)。

#### 要点：

（ 1 ）通过右对齐进行张量广播：`torch.ones(3,4,5) + torch.ones(5)`
（ 2 ）一个概率分布的形状为 `.sample().shape == batch_shape + event_shape`
（ 3 ）一个概率分布的对数概率的形状  `.log_prob (x).shape == batch_shape`（而不是 `event_shape`！）
（ 4 ）使用 `.expand()` 实现批量样本的抽取，或者通过 `plate` 自动抽取
（ 5 ）使用 `my_dist.to_event(1)` 将某个维度声明为从属维度（即依赖其他维度）
（ 6 ）使用 `with pyro.plate('name', size):` 将某个维度声明为条件独立的
（ 7 ）所有维度必须声明为从属维度或条件独立维度
（ 8 ）尝试在左侧进行批处理，这可以让 Pyro 自动并行化
    - 使用负指数，如 `x.sum(-1)` 而不是 `x.sum(2)` 
    - 使用省略号，如 `pixel = image[..., i, j]` 
    -  如果 `i,j` 是枚举的，使用 [Vindex](http://docs.pyro.ai/en/dev/ops.html#pyro.ops.indexing.Vindex) ，例如：`pixel = Vindex(image)[..., i, j] `
（ 9 ）当使用 `pyro.plate` 的自动二次采样功能时，确保数据子采样被激活了： 
    - 方法1：通过捕获索引 `with pyro.plate(...) as i: ...`  实现手动子采样；
    - 方法2：通过 `batch =pyro.subsample(data, event_dim=...)` 实现自动子采样。
（ 10 ）在调试时，使用 [Trace.format_shapes()](http://docs.pyro.ai/en/dev/poutine.html#pyro.poutine.Trace.format_shapes) 检查迹中的所有形状。

```{code-cell} ipython3
import os
import torch
import pyro
from torch.distributions import constraints
from pyro.distributions import Bernoulli, Categorical, MultivariateNormal, Normal
from pyro.distributions.util import broadcast_shape
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.poutine as poutine
from pyro.optim import Adam

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.7.0')

# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)
```

## 1 分布的形状： `batch_shape` 和 `event_shape`

熟悉 PyTorch 的人都知道，其 `Tensor` ​​有一个 `.shape` 属性，但是 `Distribution` 有所不同，它有两个具有特殊含义的形状属性：`.batch_shape` 和 `.event_shape`。这两个属性结合起来定义了一个样本的总形状。

```py
x = d.sample()
assert x.shape == d.batch_shape + d.event_shape
```

`.batch_shape` 上的索引表示条件独立型的随机变量，而`.event_shape` 上的索引则表示从属型的随机变量（即从某个分布中抽取出来的）。由于从属型的随机变量在一起定义概率， `.log_prob()` 方法只为形状 `.event_shape` 中的每个事件生成一个数值。因此`.log_prob()`的总形状是： `batch_shape`：

```py
assert d.log_prob(x).shape == d.batch_shape
```

注意 `Distribution.sample()` 方法也可以使用 `sample_shape` 参数来索引独立同分布（ I.I.D）的随机变量：

```py
x2 = d.sample(sample_shape)
assert x2.shape == sample_shape + batch_shape + event_shape
```

总的来说，有如下关系：


```
      |      iid     | independent | dependent
------+--------------+-------------+------------
shape = sample_shape + batch_shape + event_shape
```

例如，单变量分布具有`空`的事件形状（因为每个数字都是一个独立事件）。像 `MultivariateNormal` 这样向量上的分布，则有 `len(event_shape) == 1`。 而像 `InverseWishart` 这样矩阵上的分布则具有 `len(event_shape) == 2` 。

### 1.1 示例

最简单的分布形状是一元随机变量的分布：

```{code-cell} ipython3
d = Bernoulli(0.5)
assert d.batch_shape == ()
assert d.event_shape == ()
x = d.sample()
assert x.shape == ()
assert d.log_prob(x).shape == ()
```

分布可以通过传递批参数，实现批处理，如下：

```{code-cell} ipython3
d = Bernoulli(0.5 * torch.ones(3,4))
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)
```

另外一种批方法是调用 `.expand()` 方法，这只有在参数沿最左侧维度相同时才有效。例如：

```{code-cell} ipython3
d = Bernoulli(torch.tensor([0.1, 0.2, 0.3, 0.4])).expand([3, 4])
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)
```

多元随机变量的分布具有非空的 `.event_shape` 。对于这些分布而言， `.sample()` 和 `.log_prob(x)` 的形状是不同的：

```{code-cell} ipython3
d = MultivariateNormal(torch.zeros(3), torch.eye(3, 3))
assert d.batch_shape == ()
assert d.event_shape == (3,)
x = d.sample()
assert x.shape == (3,)            # == batch_shape + event_shape
assert d.log_prob(x).shape == ()  # == batch_shape
```

### 1.2 分布的重新塑形 --- 整形

在 Pyro 中，您可以通过调用 [.to_event(n)](http://docs.pyro.ai/en/dev/distributions.html#pyro.distributions.torch_distribution.TorchDistributionMixin.to_event) 属性将单变量分布视为多元分布处理，其中 `n` 是被声明为 **从属的** 的批维度数（从右侧开始）。

```{code-cell} ipython3
d = Bernoulli(0.5 * torch.ones(3,4)).to_event(1)
assert d.batch_shape == (3,)
assert d.event_shape == (4,)
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3,)
```

当您使用 Pyro 程序时，请记住，样本具有 `batch_shape + event_shape` 的形状，而 `.log_prob(x)` 具有 `batch_shape` 的形状。同时，您需要确保能够仔细地控制 `batch_shape`，通过使用 `.to_event(n)` 修剪的方式或通过 `pyro.plate` 将维度声明为独立的方式。

### 1.3 做出依赖假设总是安全的

通常在 Pyro 中，即使某些维度实际上是独立的，我们也会将其声明为具有依赖性的从属维度，例如：

```py
x = pyro.sample("x", Normal(0, 1).expand([10]).to_event(1))
assert x.shape == (10,)
```
这很有用，原因有两个：

（1）它允许我们在 `MultivariateNormal` 分布中轻松地进行交换。
（2）它稍微简化了代码，例如在下面的场景中，我们其实可以不使用 `plate`：

```py
with pyro.plate("x_plate", 10):
    x = pyro.sample("x", Normal(0, 1))  # .expand([10]) is automatic
    assert x.shape == (10,)
```

这两个版本之间的区别在于：带有 `plate` 的版本通知 Pyro ，它可以在估计梯度时利用条件独立信息；而在第一个版本中， Pyro 必须假设它们是有依赖的（即使实际上是条件独立的）。

这类似于概率图模型中的 `d-separation`：添加边并假设变量可能相关总是安全的（即扩大了模型类），但当变量实际上相关但被假设为独立时，通常是不安全的（即缩小了模型类，导致真正的模型可能位于模型类之外，就像在平均场一样）。在实践中，Pyro 的 SVI 推理算法使用面向 `正态` 分布的重参数化梯度估计器，因此两者的梯度估计具有相同的性能。

+++

## 2 用 `plate` 来声明独立的维度

Pyro 模型可以使用上下文管理器 [pyro.plate](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate) 来声明某些维度是独立的。然后推断算法可以利用这种独立性来例如构造低方差的梯度估计器，或者在线性空间而不是指数空间中做枚举。独立维度的一个例子是小批量数据上的索引：每个数据点都应该独立于所有其他数据点。

将维度声明为独立的最简单方法是：将最右侧的那些维度声明为独立的。


```py
with pyro.plate("my_plate"):
    # within this context, batch dimension -1 is independent
```

我们推荐在调试过程中，一直提供 `size` 参数来辅助对形状的调试。

```py
with pyro.plate("my_plate", len(my_data)):
    # within this context, batch dimension -1 is independent
```

从 Pyro 0.2 开始，可以实现嵌套的 `plate` ， 例如， 如果每个像素都具有独立性：

```py
with pyro.plate("x_axis", 320):
    # within this context, batch dimension -1 is independent
    with pyro.plate("y_axis", 200):
        # within this context, batch dimensions -2 and -1 are independent
```

请注意，我们总是使用负指数（如 -2、-1）从右侧开始计数。

最后，如果你想混合和匹配多个 `plate` 时，例如，仅依赖于 `x` 的噪声、仅依赖于 `y` 的噪声、同时依赖于两者的噪声等，可以声明多个 `plate` ​​并将它们用作可重用的上下文管理器。此时， Pyro 无法自动分配维度，因此需要你提供一个 `dim` 参数（依然是右侧计数）：

```py
x_axis = pyro.plate("x_axis", 3, dim=-2)
y_axis = pyro.plate("y_axis", 2, dim=-3)
with x_axis:
    # within this context, batch dimension -2 is independent
with y_axis:
    # within this context, batch dimension -3 is independent
with x_axis, y_axis:
    # within this context, batch dimensions -3 and -2 are independent
```

让我们仔细看看`plate`内的批大小。

```{code-cell} ipython3
def model1():
    a = pyro.sample("a", Normal(0, 1))
    b = pyro.sample("b", Normal(torch.zeros(2), 1).to_event(1))
    with pyro.plate("c_plate", 2):
        c = pyro.sample("c", Normal(torch.zeros(2), 1))
    with pyro.plate("d_plate", 3):
        d = pyro.sample("d", Normal(torch.zeros(3,4,5), 1).to_event(2))
    assert a.shape == ()       # batch_shape == ()     event_shape == ()
    assert b.shape == (2,)     # batch_shape == ()     event_shape == (2,)
    assert c.shape == (2,)     # batch_shape == (2,)   event_shape == ()
    assert d.shape == (3,4,5)  # batch_shape == (3,)   event_shape == (4,5) 

    x_axis = pyro.plate("x_axis", 3, dim=-2)
    y_axis = pyro.plate("y_axis", 2, dim=-3)
    with x_axis:
        x = pyro.sample("x", Normal(0, 1))
    with y_axis:
        y = pyro.sample("y", Normal(0, 1))
    with x_axis, y_axis:
        xy = pyro.sample("xy", Normal(0, 1))
        z = pyro.sample("z", Normal(0, 1).expand([5]).to_event(1))
    assert x.shape == (3, 1)        # batch_shape == (3,1)     event_shape == ()
    assert y.shape == (2, 1, 1)     # batch_shape == (2,1,1)   event_shape == ()
    assert xy.shape == (2, 3, 1)    # batch_shape == (2,3,1)   event_shape == ()
    assert z.shape == (2, 3, 1, 5)  # batch_shape == (2,3,1)   event_shape == (5,)
    
test_model(model1, model1, Trace_ELBO())
```

通过在 `batch_shape` 和 `event_shape` 之间的边界处做对齐，来可视化每个样本点的 `.shape`，对于形状调试非常有帮助：边界右侧的维度将在 `.log_prob()` 中汇集，而左侧的维度将会留存。

```
batch dims | event dims
-----------+-----------
           |        a = sample("a", Normal(0, 1))
           |2       b = sample("b", Normal(zeros(2), 1)
           |                        .to_event(1))
           |        with plate("c", 2):
          2|            c = sample("c", Normal(zeros(2), 1))
           |        with plate("d", 3):
          3|4 5         d = sample("d", Normal(zeros(3,4,5), 1)
           |                       .to_event(2))
           |
           |        x_axis = plate("x", 3, dim=-2)
           |        y_axis = plate("y", 2, dim=-3)
           |        with x_axis:
        3 1|            x = sample("x", Normal(0, 1))
           |        with y_axis:
      2 1 1|            y = sample("y", Normal(0, 1))
           |        with x_axis, y_axis:
      2 3 1|            xy = sample("xy", Normal(0, 1))
      2 3 1|5           z = sample("z", Normal(0, 1).expand([5])
           |                       .to_event(1))
```

要实现程序中样本点形状的自动检查，你可以跟踪程序并使用 [Trace.format_shapes()](http://docs.pyro.ai/en/dev/poutine.html#pyro.poutine.Trace.format_shapes) 方法，它会为每个样本点打印三个形状：分布的形状（`site["fn"].batch_shape + site["fn"].event_shape`）、值形状（`site[ "value"].shape`) 和对数概率形状（`site["log_prob"].shape`）：

```{code-cell} ipython3
trace = poutine.trace(model1).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())
```

## 3 `plate` 内部的二次采样张量

[plate](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate) 的主要用途之一是对数据进行二次采样。这在 `plate` 中是可能的：由于数据具备（条件）独立性，因此一半数据的损失期望值，在理论上，应当是完整数据损失期望值的一半。

为了做数据的二次采样，你需要通知 Pyro 全数据集和二次采样数据集的大小，Pyro 会自动选择一个随机子集，并生成索引集。

```{code-cell} ipython3
data = torch.arange(100.)

def model2():
    mean = pyro.param("mean", torch.zeros(len(data)))
    with pyro.plate("data", len(data), subsample_size=10) as ind:
        assert len(ind) == 10    # ind is a LongTensor that indexes the subsample.
        batch = data[ind]        # Select a minibatch of data.
        mean_batch = mean[ind]   # Take care to select the relevant per-datum parameters.
        # Do stuff with batch:
        x = pyro.sample("x", Normal(mean_batch, 1), obs=batch)
        assert len(x) == 10
        
test_model(model2, guide=lambda: None, loss=Trace_ELBO())
```

## 3 允许并行枚举的广播机制

Pyro 0.2 引入了并行地枚举离散隐变量的功能。当通过 [SVI](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.svi.SVI) 学习后验时，这可以显着减少梯度估计器的方差。

要使用并行枚举，Pyro 需要分配可用于枚举的张量维。为了避免与用于 `plate` 的其他维度发生冲突，需要预计并声明即将使用的最大张量维数。这个预计值被称为 `max_plate_nesting` 并且是 [SVI](http://docs.pyro.ai/en/dev/inference_algos.html)的一个参数，该参数被简单地传递给[TraceEnum_ELBO](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.traceenum_elbo.TraceEnum_ELBO)。通常 Pyro 可以自己确定这个预计值（通过运行一次 `model` 和 `guide` 对并记录运行情况），但在动态模型结构情况下，可能需要手动声明 `max_plate_nesting`。

要了解 `max_plate_nesting` 以及 Pyro 如何为枚举分配维度，让我们从上面重新审视 `model1()`。这次我们将绘制三种类型的维度：左侧的 `enumeration dimensions` （Pyro 控制这些维度）、中间的 `batch dimensions` 和右侧的 `event dimensions` 。

+++

```
      max_plate_nesting = 3
           |<--->|
enumeration|batch|event
-----------+-----+-----
           |. . .|      a = sample("a", Normal(0, 1))
           |. . .|2     b = sample("b", Normal(zeros(2), 1)
           |     |                      .to_event(1))
           |     |      with plate("c", 2):
           |. . 2|          c = sample("c", Normal(zeros(2), 1))
           |     |      with plate("d", 3):
           |. . 3|4 5       d = sample("d", Normal(zeros(3,4,5), 1)
           |     |                     .to_event(2))
           |     |
           |     |      x_axis = plate("x", 3, dim=-2)
           |     |      y_axis = plate("y", 2, dim=-3)
           |     |      with x_axis:
           |. 3 1|          x = sample("x", Normal(0, 1))
           |     |      with y_axis:
           |2 1 1|          y = sample("y", Normal(0, 1))
           |     |      with x_axis, y_axis:
           |2 3 1|          xy = sample("xy", Normal(0, 1))
           |2 3 1|5         z = sample("z", Normal(0, 1).expand([5]))
           |     |                     .to_event(1))
```

请注意，过度配置 `max_plate_nesting=4` 总是安全的，但不能配置不足，例如  `max_plate_nesting=2` 时， Pyro 会出错。

让我们看看这在实践中是如何工作的。

```{code-cell} ipython3
@config_enumerate
def model3():
    p = pyro.param("p", torch.arange(6.) / 6)
    locs = pyro.param("locs", torch.tensor([-1., 1.]))

    a = pyro.sample("a", Categorical(torch.ones(6) / 6))
    b = pyro.sample("b", Bernoulli(p[a]))  # Note this depends on a.
    with pyro.plate("c_plate", 4):
        c = pyro.sample("c", Bernoulli(0.3))
        with pyro.plate("d_plate", 5):
            d = pyro.sample("d", Bernoulli(0.4))
            e_loc = locs[d.long()].unsqueeze(-1)
            e_scale = torch.arange(1., 8.)
            e = pyro.sample("e", Normal(e_loc, e_scale)
                            .to_event(1))  # Note this depends on d.

    #                   enumerated|batch|event dims
    assert a.shape == (         6, 1, 1   )  # Six enumerated values of the Categorical.
    assert b.shape == (      2, 1, 1, 1   )  # Two enumerated Bernoullis, unexpanded.
    assert c.shape == (   2, 1, 1, 1, 1   )  # Only two Bernoullis, unexpanded.
    assert d.shape == (2, 1, 1, 1, 1, 1   )  # Only two Bernoullis, unexpanded.
    assert e.shape == (2, 1, 1, 1, 5, 4, 7)  # This is sampled and depends on d.

    assert e_loc.shape   == (2, 1, 1, 1, 1, 1, 1,)
    assert e_scale.shape == (                  7,)
            
test_model(model3, model3, TraceEnum_ELBO(max_plate_nesting=2))
```

让我们仔细看看这些维度。首先注意，Pyro 从 `max_plate_nesting` 的右侧开始分配枚举维度：Pyro 分配维度 -3 来枚举 `a`，然后维度 -4 来枚举 `b`，然后维度 -5 来枚举 `c`，最后是维度 -6 来枚举 `d`。 接下来请注意，样本在新的枚举维度中只有范围（size > 1）。这有助于保持张量小型化且降低计算成本。 请注意，`log_prob` 形状将被广播以包含 `enumeration dimensions` 和  `batch dimensions` 形状，因此 `trace.nodes['d']['log_prob'].shape == (2, 1, 1, 1, 5, 4)` 。

我们可以绘制相似的张量维度图：

```
     max_plate_nesting = 2
            |<->|
enumeration batch event
------------|---|-----
           6|1 1|     a = pyro.sample("a", Categorical(torch.ones(6) / 6))
         2 1|1 1|     b = pyro.sample("b", Bernoulli(p[a]))
            |   |     with pyro.plate("c_plate", 4):
       2 1 1|1 1|         c = pyro.sample("c", Bernoulli(0.3))
            |   |         with pyro.plate("d_plate", 5):
     2 1 1 1|1 1|             d = pyro.sample("d", Bernoulli(0.4))
     2 1 1 1|1 1|1            e_loc = locs[d.long()].unsqueeze(-1)
            |   |7            e_scale = torch.arange(1., 8.)
     2 1 1 1|5 4|7            e = pyro.sample("e", Normal(e_loc, e_scale)
            |   |                             .to_event(1))
```
为了使用枚举语义自动检查这个模型，可以创建一个枚举 `Trace`，然后使用 [Trace.format_shapes()](http://docs.pyro.ai/en/dev/poutine.html#pyro.poutine.Trace.shpaes) ：

```{code-cell} ipython3
trace = poutine.trace(poutine.enum(model3, first_available_dim=-3)).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())
```

## 4 编写可并行的代码

编写能够正确并行化处理样本点的 Pyro 模型可能很棘手。两个技巧可能会有帮助：

一是[广播机制](http://pytorch.org/docs/master/notes/broadcasting.html) ；

二是[ellipsis 切片](http://python-reference.readthedocs.io/en/dev/docs/brackets/ellipsis.html) 。

让我们通过以下模型来看看在实践中它们是如何工作的。我们的目标是编写一个既可以使用枚举也可以不使用枚举的模型。

```{code-cell} ipython3
width = 8
height = 10
sparse_pixels = torch.LongTensor([[3, 2], [3, 5], [3, 9], [7, 1]])
enumerated = None  # set to either True or False below

def fun(observe):
    p_x = pyro.param("p_x", torch.tensor(0.1), constraint=constraints.unit_interval)
    p_y = pyro.param("p_y", torch.tensor(0.1), constraint=constraints.unit_interval)
    x_axis = pyro.plate('x_axis', width, dim=-2)
    y_axis = pyro.plate('y_axis', height, dim=-1)

    # Note that the shapes of these sites depend on whether Pyro is enumerating.
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y))
    if enumerated:
        assert x_active.shape  == (2, 1, 1)
        assert y_active.shape  == (2, 1, 1, 1)
    else:
        assert x_active.shape  == (width, 1)
        assert y_active.shape  == (height,)

    # The first trick is to broadcast. This works with or without enumeration.
    p = 0.1 + 0.5 * x_active * y_active
    if enumerated:
        assert p.shape == (2, 2, 1, 1)
    else:
        assert p.shape == (width, height)
    dense_pixels = p.new_zeros(broadcast_shape(p.shape, (width, height)))

    # The second trick is to index using ellipsis slicing.
    # This allows Pyro to add arbitrary dimensions on the left.
    for x, y in sparse_pixels:
        dense_pixels[..., x, y] = 1
    if enumerated:
        assert dense_pixels.shape == (2, 2, width, height)
    else:
        assert dense_pixels.shape == (width, height)

    with x_axis, y_axis:    
        if observe:
            pyro.sample("pixels", Bernoulli(p), obs=dense_pixels)

def model4():
    fun(observe=True)

def guide4():
    fun(observe=False)

# Test without enumeration.
enumerated = False
test_model(model4, guide4, Trace_ELBO())

# Test with enumeration.
enumerated = True
test_model(model4, config_enumerate(guide4, "parallel"),
           TraceEnum_ELBO(max_plate_nesting=2))
```

## 5 在 `pyro.plate` 内部的自动广播

请注意，在所有的`模型/引导` 定义中，我们都依赖 [pyro.plate](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate) 自动扩展样本形状以满足由 `pyro.sample` 语句强制执行的批形状约束。不过，这种广播机制等效于手动说明的 `.expand()` 语句。

我们将使用 [上一节](#Writing-parallelizable-code) 中的 `model4` 来演示这一点。需要对之前代码的以下更改： 

- 出于本示例的目的，将仅考虑 `并行` 枚举，但广播应该在没有枚举或使用 `顺序` 枚举的情况下按预期工作。 

- 我们已经分离出采样函数，该函数返回与活动像素对应的张量。将模型代码模块化为组件是一种常见做法，有助于大型模型的可维护性。 

- 我们还想使用 `pyro.plate` 能够构造在 [num_particles](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.elbo.ELBO) 上并行化的 ELBO 估计器。这是通过将 `模型/引导` 的内容包装在最外层 `pyro.plate` 的上下文中来完成的。

```{code-cell} ipython3
num_particles = 100  # Number of samples for the ELBO estimator
width = 8
height = 10
sparse_pixels = torch.LongTensor([[3, 2], [3, 5], [3, 9], [7, 1]])

def sample_pixel_locations_no_broadcasting(p_x, p_y, x_axis, y_axis):
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x).expand([num_particles, width, 1]))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y).expand([num_particles, 1, height]))
    return x_active, y_active

def sample_pixel_locations_full_broadcasting(p_x, p_y, x_axis, y_axis):
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y))
    return x_active, y_active 

def sample_pixel_locations_partial_broadcasting(p_x, p_y, x_axis, y_axis):
    with x_axis:
        x_active = pyro.sample("x_active", Bernoulli(p_x).expand([width, 1]))
    with y_axis:
        y_active = pyro.sample("y_active", Bernoulli(p_y).expand([height]))
    return x_active, y_active 

def fun(observe, sample_fn):
    p_x = pyro.param("p_x", torch.tensor(0.1), constraint=constraints.unit_interval)
    p_y = pyro.param("p_y", torch.tensor(0.1), constraint=constraints.unit_interval)
    x_axis = pyro.plate('x_axis', width, dim=-2)
    y_axis = pyro.plate('y_axis', height, dim=-1)

    with pyro.plate("num_particles", 100, dim=-3):
        x_active, y_active = sample_fn(p_x, p_y, x_axis, y_axis)
        # Indices corresponding to "parallel" enumeration are appended 
        # to the left of the "num_particles" plate dim.
        assert x_active.shape  == (2, 1, 1, 1)
        assert y_active.shape  == (2, 1, 1, 1, 1)
        p = 0.1 + 0.5 * x_active * y_active
        assert p.shape == (2, 2, 1, 1, 1)

        dense_pixels = p.new_zeros(broadcast_shape(p.shape, (width, height)))
        for x, y in sparse_pixels:
            dense_pixels[..., x, y] = 1
        assert dense_pixels.shape == (2, 2, 1, width, height)

        with x_axis, y_axis:    
            if observe:
                pyro.sample("pixels", Bernoulli(p), obs=dense_pixels)

def test_model_with_sample_fn(sample_fn):
    def model():
        fun(observe=True, sample_fn=sample_fn)

    @config_enumerate
    def guide():
        fun(observe=False, sample_fn=sample_fn)

    test_model(model, guide, TraceEnum_ELBO(max_plate_nesting=3))

test_model_with_sample_fn(sample_pixel_locations_no_broadcasting)
test_model_with_sample_fn(sample_pixel_locations_full_broadcasting)
test_model_with_sample_fn(sample_pixel_locations_partial_broadcasting)
```

在第一个采样函数中，我们必须进行一些手动记录并扩展 `伯努利` 分布的批形状，以应对由 `pyro.plate` 上下文添加的条件独立性维度。特别要注意， `sample_pixel_locations` 需要有关`num_particles`、`width` 和`height` 的知识，并且会从全局范围访问这些变量，这并不理想。

- 需要提供 `pyro.plate` 的第二个参数，即可选的 `size` 参数，用于隐式广播，以便可以推断每个样本点的批形状要求。 

- 样本点的现有 `batch_shape` 必须能够以 `pyro.plate` 上下文中的大小进行广播。在此处的特定示例中，`Bernoulli(p_x)` 有一个空的批形状，它是普遍可广播的。

请注意使用 `pyro.plate` 并通过张量化操作实现并行化是多么简单！ `pyro.plate` 还有助于代码模块化，因为模型组件可以编写为与 `plate` 上下文无关，它们随后可能会嵌入其中。

```{code-cell} ipython3

```
