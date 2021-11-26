---
jupytext:
  formats: ipynb,md:myst
  text_representation: {extension: .md, format_name: myst, format_version: 0.13, jupytext_version: 1.13.1}
kernelspec: {display_name: Python 3, language: python, name: python3}
---

# Modules in Pyro


This tutorial introduces [PyroModule](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroModule), Pyro's Bayesian extension of PyTorch's [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) class. Before starting you should understand the basics of Pyro [models](http://pyro.ai/examples/intro_part_i.html) and [inference](http://pyro.ai/examples/intro_part_ii.html), understand the two primitives [pyro.sample()](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.sample) and [pyro.param()](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.param), and understand the basics of Pyro's [effect handlers](http://pyro.ai/examples/effect_handlers.html) (e.g. by browsing [minipyro.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/minipyro.py)).

#### Summary:

- [PyroModule](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroModule)s are like [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)s but allow Pyro effects for sampling and constraints.
- `PyroModule` is a mixin subclass of `nn.Module` that overrides attribute access (e.g. `.__getattr__()`).
- There are three different ways to create a `PyroModule`:
  - create a new subclass: `class MyModule(PyroModule): ...`,
  - Pyro-ize an existing class: `MyModule = PyroModule[OtherModule]`, or
  - Pyro-ize an existing `nn.Module` instance in-place: `to_pyro_module_(my_module)`.
- Usual `nn.Parameter` attributes of a `PyroModule` become Pyro parameters.
- Parameters of a `PyroModule` synchronize with Pyro's global param store.
- You can add constrained parameters by creating [PyroParam](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroParam) objects.
- You can add stochastic attributes by creating [PyroSample](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroSample) objects.
- Parameters and stochastic attributes are named automatically (no string required).
- `PyroSample` attributes are sampled once per `.__call__()` of the outermost `PyroModule`.
- To enable Pyro effects on methods other than `.__call__()`, decorate them with @[pyro_method](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.pyro_method).
- A `PyroModule` model may contain `nn.Module` attributes.
- An `nn.Module` model may contain at most one `PyroModule` attribute (see [naming section](#Caution-avoiding-duplicate-names)).
- An `nn.Module` may contain both a `PyroModule` model and `PyroModule` guide (e.g. [Predictive](http://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.predictive.Predictive)).

#### Table of Contents

- [How PyroModule works](#How-PyroModule-works)
- [How to create a PyroModule](#How-to-create-a-PyroModule)
- [How effects work](#How-effects-work)
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

## How `PyroModule` works  <a class="anchor" id="How-PyroModule-works"></a>

[PyroModule](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroModule) aims to combine Pyro's primitives and effect handlers with PyTorch's [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) idiom, thereby enabling Bayesian treatment of existing `nn.Module`s and enabling model serving via [jit.trace_module](https://pytorch.org/docs/stable/jit.html#torch.jit.trace_module). Before you start using `PyroModule`s it will help to understand how they work, so you can avoid pitfalls.

`PyroModule` is a subclass of `nn.Module`. `PyroModule` enables Pyro effects by inserting effect handling logic on module attribute access, overriding the `.__getattr__()`, `.__setattr__()`, and `.__delattr__()` methods. Additionally, because some effects (like sampling) apply only once per model invocation, `PyroModule` overrides the `.__call__()` method to ensure samples are generated at most once per `.__call__()` invocation (note `nn.Module` subclasses typically implement a `.forward()` method that is called by `.__call__()`).

+++

## How to create a `PyroModule`   <a class="anchor" id="How-to-create-a-PyroModule"></a>

There are three ways to create a `PyroModule`. Let's start with a `nn.Module` that is not a `PyroModule`:

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

The first way to create a `PyroModule` is to create a subclass of `PyroModule`. You can update any `nn.Module` you've written to be a PyroModule, e.g.
```diff
- class Linear(nn.Module):
+ class Linear(PyroModule):
      def __init__(self, in_size, out_size):
          super().__init__()
          self.weight = ...
          self.bias = ...
      ...
```
Alternatively if you want to use third-party code like the `Linear` above you can subclass it, using `PyroModule` as a mixin class

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

The second way to create a `PyroModule` is to use bracket syntax `PyroModule[-]` to automatically denote a trivial mixin class as above.
```diff
- linear = Linear(5, 2)
+ linear = PyroModule[Linear](5, 2)
```
In our case we can write

```{code-cell} ipython3
linear = PyroModule[Linear](5, 2)
assert isinstance(linear, nn.Module)
assert isinstance(linear, Linear)
assert isinstance(linear, PyroModule)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

The one difference between manual subclassing and using `PyroModule[-]` is that `PyroModule[-]` also ensures all `nn.Module` superclasses also become `PyroModule`s, which is important for class hierarchies in library code. For example since `nn.GRU` is a subclass of `nn.RNN`, also `PyroModule[nn.GRU]` will be a subclass of `PyroModule[nn.RNN]`.

The third way to create a `PyroModule` is to change the type of an existing `nn.Module` instance in-place using [to_pyro_module_()](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.to_pyro_module_). This is useful if you're using a third-party module factory helper or updating an existing script, e.g.

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

## How effects work <a class="anchor" id="How-effects-work"></a>

So far we've created `PyroModule`s but haven't made use of Pyro effects. But already the `nn.Parameter` attributes of our `PyroModule`s act like [pyro.param](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.param) statements: they synchronize with Pyro's param store, and they can be recorded in traces.

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

## How to constrain parameters  <a class="anchor" id="How-to-constrain-parameters"></a>

Pyro parameters allow constraints, and often we want our `nn.Module` parameters to obey constraints. You can constrain a `PyroModule`'s parameters by replacing `nn.Parameter` with a [PyroParam](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroParam) attribute. For example to ensure the `.bias` attribute is positive, we can set it to

```{code-cell} ipython3
print("params before:", [name for name, _ in linear.named_parameters()])

linear.bias = PyroParam(torch.randn(2).exp(), constraint=constraints.positive)
print("params after:", [name for name, _ in linear.named_parameters()])
print("bias:", linear.bias)

example_input = torch.randn(100, 5)
example_output = linear(example_input)
assert example_output.shape == (100, 2)
```

Now PyTorch will optimize the `.bias_unconstrained` parameter, and each time we access the `.bias` attribute it will read and transform the `.bias_unconstrained` parameter (similar to a Python `@property`).


If you know the constraint beforehand, you can build it into the module constructor, e.g.
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

## How to make a `PyroModule`  Bayesian  <a class="anchor" id="How-to-make-a-PyroModule-Bayesian"></a>

So far our `Linear` module is still deterministic. To make it randomized and Bayesian, we'll replace `nn.Parameter` and `PyroParam` attributes with [PyroSample](http://docs.pyro.ai/en/stable/nn.html#pyro.nn.module.PyroSample) attributes, specifying a prior. Let's put a simple prior over the weights, taking care to expand its shape to `[5,2]` and declare event dimensions with [.to_event()](http://docs.pyro.ai/en/stable/distributions.html#pyro.distributions.torch_distribution.TorchDistributionMixin.to_event) (as explained in the [tensor shapes tutorial](https://pyro.ai/examples/tensor_shapes.html)).

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

Notice that the `.weight` parameter now disappears, and each time we call `linear()` a new weight is sampled from the prior. In fact, the weight is sampled when the `Linear.forward()` accesses the `.weight` attribute: this attribute now has the special behavior of sampling from the prior.

We can see all the Pyro effects that appear in the trace:

```{code-cell} ipython3
with poutine.trace() as tr:
    linear(example_input)
for site in tr.trace.nodes.values():
    print(site["type"], site["name"], site["value"])
```

So far we've modified a third-party module to be Bayesian
```py
linear = Linear(...)
to_pyro_module_(linear)
linear.bias = PyroParam(...)
linear.weight = PyroSample(...)
```
If you are creating a model from scratch, you could instead define a new class

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

Note that samples are drawn at most once per `.__call__()` invocation, for example
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

## ⚠ Caution: accessing attributes inside plates  <a class="anchor" id="⚠-Caution:-accessing-attributes-inside-plates"></a>

Because `PyroSample` and `PyroParam` attributes are modified by Pyro effects, we need to take care where parameters are accessed. For example [pyro.plate](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.plate) contexts can change the shape of sample and param sites. Consider a model with one latent variable and a batched observation statement. We see that the only difference between these two models is where the `.loc` attribute is accessed.

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

## How to create a complex nested `PyroModule` <a class="anchor" id="How-to-create-a-complex-nested-PyroModule"></a>

To perform inference with the above `BayesianLinear` module we'll need to wrap it in probabilistic model with a likelihood; that wrapper will also be a `PyroModule`.

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

Whereas a usual `nn.Module` can be trained with a simple PyTorch optimizer, a Pyro model requires probabilistic inference, e.g. using [SVI](http://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.svi.SVI) and an [AutoNormal](http://docs.pyro.ai/en/stable/infer.autoguide.html#pyro.infer.autoguide.AutoNormal) guide. See the [bayesian regression tutorial](http://pyro.ai/examples/bayesian_regression.html) for details.

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

`PyroSample` statements may also depend on other sample statements or parameters. In this case the `prior` can be a callable depending on `self`, rather than a constant distribution. For example consider the hierarchical model

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

## How naming works  <a class="anchor" id="How-naming-works"></a>

In the above code we saw a `BayesianLinear` model embedded inside another `Model`. Both were `PyroModule`s. Whereas simple [pyro.sample](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.sample) statements require name strings, `PyroModule` attributes handle naming automatically. Let's see how that works with the above `model` and `guide` (since `AutoNormal` is also a `PyroModule`).

Let's trace executions of the model and the guide.

```{code-cell} ipython3
with poutine.trace() as tr:
    model(x)
for site in tr.trace.nodes.values():
    print(site["type"], site["name"], site["value"].shape)
```

Observe that `model.linear.bias` corresponds to the `linear.bias` name, and similarly for the `model.linear.weight` and `model.obs_scale` attributes. The "instances" site corresponds to the plate, and the "obs" site corresponds to the likelihood. Next examine the guide:

```{code-cell} ipython3
with poutine.trace() as tr:
    guide(x)
for site in tr.trace.nodes.values():
    print(site["type"], site["name"], site["value"].shape)
```

We see the guide learns posteriors over three random variables: `linear.bias`, `linear.weight`, and `obs_scale`. For each of these, the guide learns a `(loc,scale)` pair of parameters, which are stored internally in nested `PyroModules`:
```python
class AutoNormal(...):
    def __init__(self, ...):
        self.locs = PyroModule()
        self.scales = PyroModule()
        ...
```
Finally, `AutoNormal` contains a `pyro.sample` statement for each unconstrained latent site followed by a [pyro.deterministic](http://docs.pyro.ai/en/stable/primitives.html#pyro.primitives.deterministic) statement to map the unconstrained sample to a constrained posterior sample.

+++

## ⚠ Caution: avoiding duplicate names <a class="anchor" id="⚠-Caution:-avoiding-duplicate-names"></a>

`PyroModule`s name their attributes automatically, event for attributes nested deeply in other `PyroModule`s. However care must be taken when mixing usual `nn.Module`s with `PyroModule`s, because `nn.Module`s do not support automatic site naming.

Within a single model (or guide):

If there is only a single `PyroModule`, then your are safe.
```diff
  class Model(nn.Module):        # not a PyroModule
      def __init__(self):
          self.x = PyroModule()
-         self.y = PyroModule()  # Could lead to name conflict.
+         self.y = nn.Module()  # Has no Pyro names, so avoids conflict.
```
If there are only two `PyroModule`s then one must be an attribute of the other.
```diff
class Model(PyroModule):
    def __init__(self):
       self.x = PyroModule()  # ok
```
If you have two `PyroModule`s that are not attributes of each other, then they must be connected by attribute links through other `PyroModule`s. These can be sibling links
```diff
- class Model(nn.Module):     # Could lead to name conflict.
+ class Model(PyroModule):    # Ensures names are unique.
      def __init__(self):
          self.x = PyroModule()
          self.y = PyroModule()
```
or ancestor links
```diff
  class Model(PyroModule):
      def __init__(self):
-         self.x = nn.Module()    # Could lead to name conflict.
+         self.x = PyroModule()   # Ensures y is conected to root Model.
          self.x.y = PyroModule()
```

Sometimes you may want to store a `(model,guide)` pair in a single `nn.Module`, e.g. to serve them from C++. In this case it is safe to make them attributes of a container `nn.Module`, but that container should *not* be a `PyroModule`.
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
