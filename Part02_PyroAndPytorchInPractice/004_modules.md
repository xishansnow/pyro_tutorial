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

# Pyro 中的模块

本教程主要介绍 [PyroModule](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroModule)，[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 类的一个 Pyro 贝叶斯扩展。

在开始之前你应该了解关于 Pyro 中的 [模型](http://pyro.ai/examples/intro_part_i.html)和[推断](http://pyro.ai/examples/intro_part_ii.html) 基础知识，了解 [pyro.sample()](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.sample) 和 [pyro.param()](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.param) 两个基本元语，并了解 [Pyro Effects 处理程序](http://pyro.ai/examples/Effects_handlers.html) 的基础知识（例如通过浏览 [minipyro.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/minipyro.py) ）。

#### 要点:

- [PyroModule](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroModule) 就像 [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) ，但允许  Pyro 的采样和约束等 Effects 。 

- `PyroModule` 是 `nn.Module` 的混合子类，它重载了属性访问（例如 `.__getattr__()`）。 
- 创建自定义的 `PyroModule` 有三种不同的方法：
    - 创建一个 `PyroModule` 的子类：`class MyModule(PyroModule): ...`，
    - Pyro 化现有的类：`MyModule = PyroModule[OtherModule]`
    - 就地 Pyro 化一个现有的 `nn.Module` 实例：`to_pyro_module_(my_module)`。
- 通常`PyroModule` 的`nn.Parameter` 属性都变成了 Pyro 参数。
- `PyroModule` 的参数与 Pyro 的全局参数存储库同步。
- 您可以通过创建 [PyroParam 对象](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroParam) 来添加约束参数。
- 您可以通过创建 [PyroSample 对象](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroSample) 来添加随机属性。
- 参数和随机属性会自动命名（不需要字符串）。
- 最外层`PyroModule` 的每此 `.__call__()` ，都会对 `PyroSample` 属性做一次采样。
- 要想在 `.__call__()` 以外的方法中启用 Pyro  Effects ， 应当用 [@pyro_method](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.pyro_method) 来装饰它们。
- 一个 `PyroModule` 模型可能包含 `nn.Module` 属性。
- 一个 `nn.Module` 模型最多只能包含一个 `PyroModule` 属性，参见 [命名部分](#Caution-avoiding-duplicate-names) 。
- `nn.Module` 可能既包含一个 `PyroModule` 模型，又包含 `PyroModule` 引导，例如 [预测性分布](http://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.predictive.Predictive) 。

#### Table of Contents

- [How PyroModule works](#How-PyroModule-works)
- [How to create a PyroModule](#How-to-create-a-PyroModule)
- [How Effects work](#How-Effects-work)
- [How to constrain parameters](#How-to-constrain-parameters)
- [How to make a PyroModule Bayesian](#How-to-make-a-PyroModule-Bayesian)
- [Caution: accessing attributes inside plates](#⚠-Caution:-accessing-attributes-inside-plates)
- [How to create a complex nested PyroModule](#How-to-create-a-complex-nested-PyroModule)
- [How naming works](#How-naming-works)
- [Caution: avoiding duplicate names](#⚠-Caution:-avoiding-duplicate-names)

```{code-cell} ipython3
import os
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from torch.distributions import constraints
from pyro.nn import PyroModule, PyroParam, PyroSample
from pyro.nn.module import to_pyro_module_
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.7.0')
```

## 1 `PyroModule` 的工作原理

[PyroModule](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroModule) 旨在将 Pyro 的元语和 Effects 处理程序与 PyTorch 的 [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 习惯用法相结合，从而实现对现有 `nn.Module` 的贝叶斯处理，并通过  [j​​it.trace_module](https://pytorch.org/docs/stable/jit.html#torch.jit.trace_module) 实现模型服务。在开始使用 `PyroModule`s 之前，了解它们的工作原理将有助于避免陷阱。


`PyroModule` 是 `nn.Module` 的子类。 `PyroModule` 通过在模块属性访问中插入 Effects 处理逻辑来实现 Pyro Effects，如：重载 `.__getattr__()`、`.__setattr__()` 和 `.__delattr__()` 方法来启用 Pyro Effects。

此外，由于某些 Effects （如采样）仅在每次模型调用时应用一次，`PyroModule` 重载了 `.__call__()` 方法以确保每次 `.__call__()` 调用最多只生成一次样本。注意： `nn.Module ` 子类通常实现一个由 `.__call__()` 调用的 `.forward()` 方法。

+++

## 2 如何创建一个 `PyroModule`

有三种创建 `PyroModule` 的途径。 首先从并非 `PyroModule` 的 `nn.Module` 开始：

```{code-cell} ipython3
class Linear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_size, out_size))
        self.bias = nn.Parameter(torch.randn(out_size))
        
    def forward(self, input_):
        return self.bias + input_ @ self.weight
    
linear = Linear(5, 2)
assert isinstance(linear, nn.Module)
assert not isinstance(linear, PyroModule)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

**（1） 创建 `PyroModule` 的子类**

第一个创建 `PyroModule` 的方法是创建一个 `PyroModule` 的子类。你可以将所有的 `nn.Module` 改写成一个 `PyroModule` ，例如：

```diff
- class Linear(nn.Module):
+ class Linear(PyroModule):
      def __init__(self, in_size, out_size):
          super().__init__()
          self.weight = ...
          self.bias = ...
      ...
```

**（2） 将 `PyroModule` 作为混合类**

当你想使用第三方代码（如上例中的 `Linear`）时，可以创建一个该类的子类，并将 `PyroModule` 作为混合类（ mixin class ）：

```{code-cell} ipython3
class PyroLinear(Linear, PyroModule):
    pass

linear = PyroLinear(5, 2)
assert isinstance(linear, nn.Module)
assert isinstance(linear, Linear)
assert isinstance(linear, PyroModule)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

`PyroModule` 另外提供了一种 `PyroModule[-]` 自动化语法来实现混合类。

```diff
- linear = Linear(5, 2)
+ linear = PyroModule[Linear](5, 2)
```

在我们的示例中，可以写为：

```{code-cell} ipython3
linear = PyroModule[Linear](5, 2)
assert isinstance(linear, nn.Module)
assert isinstance(linear, Linear)
assert isinstance(linear, PyroModule)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

两者之间（ 即用子类手工定义和用`[]`语法自动定义 ）的一个区别是： `PyroModule[-]` 方法可以确保所有 `nn.Module` 超类也变成了 `PyroModule`， 这对于库代码中的类层次结构非常重要。 例如： 由于 `nn.GRU` 是一个 `nn.RNN` 的子类，因此 `PyroModule[nn.GRU]` 也会是 `PyroModule[nn.RNN]` 的子类。

**（3）使用 `to_pyro_module_()` 将 `nn.Module` 转换为 `PyroModule`**

第三种方法是使用 [`to_pyro_module_()`](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.to_pyro_module_) 方法将 `nn.Module` 就地转换为 `PyroModule` 。当你使用第三方模块工程 helper 或更新已有脚本时，会非常有用。 例如：

```{code-cell} ipython3
linear = Linear(5, 2)
assert isinstance(linear, nn.Module)
assert not isinstance(linear, PyroModule)

to_pyro_module_(linear)  # this operates in-place
assert isinstance(linear, nn.Module)
assert isinstance(linear, Linear)
assert isinstance(linear, PyroModule)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

## 3 Effects 是如何工作的？

虽然创建了 `PyroModule` ，但还没有使用 `Pyro  Effects` 。不过我们已经介绍过 `PyroModle` 的 `nn.Parameter` 属性行为类似于 [pyro.param](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.param) 语句: 这些参数将与 Pyro 的参数存储库保持同步，并记录在迹中。

```{code-cell} ipython3
pyro.clear_param_store()

# This is not traced:
linear = Linear(5, 2)
with poutine.trace() as tr:
    linear(example_input)
print(type(linear).__name__)
print(list(tr.trace.nodes.keys()))
print(list(pyro.get_param_store().keys()))

# Now this is traced:
to_pyro_module_(linear)
with poutine.trace() as tr:
    linear(example_input)
print(type(linear).__name__)
print(list(tr.trace.nodes.keys()))
print(list(pyro.get_param_store().keys()))
```

## 4 如何约束参数？

Pyro 参数允许设置约束，而且我们经常希望 `nn.Module` 参数遵循某种约束。 此时，你可以通过将 `nn.Parameter` 替换为 [PyroParam](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroParam) 属性来约束 `PyroModule` 的参数。 例如为确保 `.bias` 参数的属性保持为正实数，我们可以用以下方法设置：

```{code-cell} ipython3
print("params before:", [name for name, _ in linear.named_parameters()])

linear.bias = PyroParam(torch.randn(2).exp(), constraint=constraints.positive)
print("params after:", [name for name, _ in linear.named_parameters()])
print("bias:", linear.bias)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

现在 PyTorch 将对 `.bias_unconstrained` 参数进行优化， 并且每次我们访问 `.bias` 的属性时，它将自动读取和转换 `.bias_unconstrained` 参数 ( 类似于 Python 的 `@property` ).

如果你预先知道约束，你可以将在 module 的构造器中创建它，例如：

```diff
  class Linear(PyroModule):
      def __init__(self, in_size, out_size):
          super().__init__()
          self.weight = ...
-         self.bias = nn.Parameter(torch.randn(out_size))
+         self.bias = PyroParam(torch.randn(out_size).exp(),
+                               constraint=constraints.positive)
      ...
```

+++

## 5 如何使 `PyroModule` 贝叶斯化

然而我们的 `Linear` 模块依然是确定性的。 为了使其具备随机性并且是贝叶斯的， 在声明一个先验同时， 需要将 `nn.Parameter` 和 `PyroParam` 属性替换为 [PyroSample 属性](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroSample) 。 让我们在权重上配置一个简单先验，注意将其形状扩展为 `[5,2]` ，并且用 [.to_event()](http://docs.pyro.ai/en/stable/distributions.html#pyro.distributions.torch_distribution.TorchDistributionMixin.to_event) 声明 event 维 ( 参见 [tensor shapes tutorial](https://pyro.ai/examples/tensor_shapes.html) )。

```{code-cell} ipython3
print("params before:", [name for name, _ in linear.named_parameters()])

linear.weight = PyroSample(dist.Normal(0, 1).expand([5, 2]).to_event(2))
print("params after:", [name for name, _ in linear.named_parameters()])
print("weight:", linear.weight)
print("weight:", linear.weight)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

注意 `.weight` 参数消失了，并且每次调用 `linear()` 时，会从先验中采样得到一个新的权重。 事实上，权重的采样是发生在 `Linear.forward()` 访问 `.weight` 属性的时候： 该属性现在有了从先验中采样的特殊行为。

我们可以看看迹中出现的所有 Pyro Effects ：

```{code-cell} ipython3
with poutine.trace() as tr:
    linear(example_input)
for site in tr.trace.nodes.values():
    print(site["type"], site["name"], site["value"])
```

到此我们已经将一个第三方的模块转换成贝叶斯的了。

```py
linear = Linear(...)
to_pyro_module_(linear)
linear.bias = PyroParam(...)
linear.weight = PyroSample(...)
```
如果你从头创建一个模型，则你可以定义一个新类：

```{code-cell} ipython3
class BayesianLinear(PyroModule):
    def __init__(self, in_size, out_size):
       super().__init__()
       self.bias = PyroSample(
           prior=dist.LogNormal(0, 1).expand([out_size]).to_event(1))
       self.weight = PyroSample(
           prior=dist.Normal(0, 1).expand([in_size, out_size]).to_event(2))

    def forward(self, input):
        return self.bias + input @ self.weight  # this line samples bias and weight
```

注意，在每次调用 `.__call__()` 时只能抽取最多一个样本，例如：

```py
class BayesianLinear(PyroModule):
    ...
    def forward(self, input):
        weight1 = self.weight      # Draws a sample.
        weight2 = self.weight      # Reads previous sample.
        assert weight2 is weight1  # All accesses should agree.
        ...
```

+++

## 6 提醒: 在 `plates` 内访问属性

因为 `PyroSample` 和 `PyroParam` 属性是被 Pyro Effects 修改的，因此我们需要注意参数是在哪里被访问的。例如： [pyro.plate](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.plate) 上下文可以改变样本和参数点的形状。考虑一个具有一个隐变量和一批观测量的模型，我们可以看到两种模型之间唯一的区别就是 `.loc` 属性被访问了。

```{code-cell} ipython3
class NormalModel(PyroModule):
    def __init__(self):
        super().__init__()
        self.loc = PyroSample(dist.Normal(0, 1))

class GlobalModel(NormalModel):
    def forward(self, data):
        # If .loc is accessed (for the first time) outside the plate,
        # then it will have empty shape ().
        loc = self.loc
        assert loc.shape == ()
        with pyro.plate("data", len(data)):
            pyro.sample("obs", dist.Normal(loc, 1), obs=data)
        
class LocalModel(NormalModel):
    def forward(self, data):
        with pyro.plate("data", len(data)):
            # If .loc is accessed (for the first time) inside the plate,
            # then it will be expanded by the plate to shape (plate.size,).
            loc = self.loc
            assert loc.shape == (len(data),)
            pyro.sample("obs", dist.Normal(loc, 1), obs=data)

data = torch.randn(10)
LocalModel()(data)
GlobalModel()(data)
```

## 7 如何创建复杂的嵌套 `PyroModule` 

为了使用上述 `BayesianLinear` 模块进行贝叶斯推断，我们需要在概率模型中为其指定一个似然，该似然应当也被封装为`PyroModule` 。

```{code-cell} ipython3
class Model(PyroModule):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = BayesianLinear(in_size, out_size)  # this is a PyroModule
        self.obs_scale = PyroSample(dist.LogNormal(0, 1))

    def forward(self, input, output=None):
        obs_loc = self.linear(input)  # this samples linear.bias and linear.weight
        obs_scale = self.obs_scale    # this samples self.obs_scale
        with pyro.plate("instances", len(input)):
            return pyro.sample("obs", dist.Normal(obs_loc, obs_scale).to_event(1),
                               obs=output)
```

就像一般的 `nn.Module` 都能够使用简单的 PyTorch 优化器进行训练一样， 一个 Pyro 模型也可以使用 [SVI](http://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.svi.SVI) 和 [AutoNormal](http://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoNormal) 等引导函数做概率推断。细节参见 [bayesian regression tutorial](http://pyro.ai/examples/bayesian_regression.html) 。

```{code-cell} ipython3
%%time
pyro.clear_param_store()
pyro.set_rng_seed(1)

model = Model(5, 2)
x = torch.randn(100, 5)
y = model(x)

guide = AutoNormal(model)
svi = SVI(model, guide, Adam({"lr": 0.01}), Trace_ELBO())
for step in range(2 if smoke_test else 501):
    loss = svi.step(x, y) / y.numel()
    if step % 100 == 0:
        print("step {} loss = {:0.4g}".format(step, loss))
```

`PyroSample` 语句可以依赖于其他 `sample` 语句或参数。在本示例中，先验 `prior` 是一个依赖于自身的可调用对象，而不是一个常值分布。 例如，考虑如下分层模型：

```{code-cell} ipython3
class Model(PyroModule):
    def __init__(self):
        super().__init__()
        self.dof = PyroSample(dist.Gamma(3, 1))
        self.loc = PyroSample(dist.Normal(0, 1))
        self.scale = PyroSample(lambda self: dist.InverseGamma(self.dof, 1))
        self.x = PyroSample(lambda self: dist.Normal(self.loc, self.scale))
        
    def forward(self):
        return self.x
    
Model()()
```

## 8 命名是如何工作的？

在上面代码中，我们看到了一个 `BayesianLinear` 模型，其中嵌入了另外一个 `Model`，两者都是 `PyroModule`。与简单的 [pyro.sample](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.sample) 语句需要名称字符串作为参数一样， `PyroModule` 属性也会自动处理命名。让我们看下在上述 `model` 和 `guide` 中它们是如何工作的 ( 因为 `AutoNormal` 也是一个 `PyroModule`).

让我们跟踪该`模型`和`引导`的执行。

```{code-cell} ipython3
with poutine.trace() as tr:
    model(x)
for site in tr.trace.nodes.values():
    print(site["type"], site["name"], site["value"].shape)
```

可以观对应于 `linear.bias` 名称的 `model.linear.bias` 属性，以及类似的 `model.linear.weight` 和 `model.obs_scale` 属性。 相应的实例对应于 `plate` 和对应于似然的 `obs` 观测数据点。下一步检查`引导`：

```{code-cell} ipython3
with poutine.trace() as tr:
    guide(x)
for site in tr.trace.nodes.values():
    print(site["type"], site["name"], site["value"].shape)
```

可以看到 `引导` 学习三个随机变量上的后验分布，它们分别是: `linear.bias`, `linear.weight`, 和 `obs_scale`。对于其中每一个， `引导` 学习了一个保存在嵌套 `PyroModules` 内部的 `(loc,scale)` 参数对：

```python
class AutoNormal(...):
    def __init__(self, ...):
        self.locs = PyroModule()
        self.scales = PyroModule()
        ...
```

最终， `AutoNormal` 为每一个后跟了 [pyro.deterministic](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.deterministic) 语句的无约束隐变量包含了一个 `pyro.sample` 语句，用于将无约束样本映射到有约束的后验样本。

+++

## 9 注意：避免重名

`PyroModule` 自动命名它们的属性，用于嵌套在其他 `PyroModule` 中的属性的`event`。然而，在混合 `nn.Module` 和 `PyroModule` 时必须小心，因为 `nn.Module` 不支持自动命名。

在一个简单的`模型`( 或 `引导` ) 中：

如果仅有一个 `PyroModule`，则你是安全的。

```diff
  class Model(nn.Module):        # not a PyroModule
      def __init__(self):
          self.x = PyroModule()
-         self.y = PyroModule()  # Could lead to name conflict.
+         self.y = nn.Module()  # Has no Pyro names, so avoids conflict.
```
如果仅有两个 `PyroModule`，那么其中一个必须是另外一个的属性之一。

```diff
class Model(PyroModule):
    def __init__(self):
       self.x = PyroModule()  # ok
```

如果你有两个不是彼此属性的 `PyroModule`，那么它们必须通过其他 `PyroModule` 的属性进行链接。

下面是通过兄弟的链接：

```diff
- class Model(nn.Module):     # Could lead to name conflict.
+ class Model(PyroModule):    # Ensures names are unique.
      def __init__(self):
          self.x = PyroModule()
          self.y = PyroModule()
```

或者祖先链接：

```diff
  class Model(PyroModule):
      def __init__(self):
-         self.x = nn.Module()    # Could lead to name conflict.
+         self.x = PyroModule()   # Ensures y is conected to root Model.
          self.x.y = PyroModule()
```

有时你可能想在单个 `nn.Module` 中存储一个 `(model,guide)` 对。此时使它们成为容器 `nn.Module` 的属性是安全的，但该容器不应该是`PyroModule`。

```python
class Container(nn.Module):            # This cannot be a PyroModule.
    def __init__(self, model, guide):  # These may be PyroModules.
        super().__init__()
        self.model = model
        self.guide = guide
    # This is a typical trace-replay pattern seen in model serving.
    def forward(self, data):
        tr = poutine.trace(self.guide).get_trace(data)
        return poutine.replay(model, tr)(data)
```

```{code-cell} ipython3

```
