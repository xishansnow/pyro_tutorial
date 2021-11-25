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

# Inference with Discrete Latent Variables

This tutorial describes Pyro's enumeration strategy for discrete latent variable models.
This tutorial assumes the reader is already familiar with the [Tensor Shapes Tutorial](http://pyro.ai/examples/tensor_shapes.html).

#### Summary 

- Pyro implements automatic enumeration over discrete latent variables.
- This strategy can be used alone or inside SVI (via [TraceEnum_ELBO](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.traceenum_elbo.TraceEnum_ELBO)), HMC, or NUTS.
- The standalone [infer_discrete](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.discrete.infer_discrete) can generate samples or MAP estimates.
- Annotate a sample site `infer={"enumerate": "parallel"}` to trigger enumeration.
- If a sample site determines downstream structure, instead use `{"enumerate": "sequential"}`.
- Write your models to allow arbitrarily deep batching on the left, e.g. use broadcasting.
- Inference cost is exponential in treewidth, so try to write models with narrow treewidth.
- If you have trouble, ask for help on [forum.pyro.ai](https://forum.pyro.ai)!

#### Table of contents

- [Overview](#Overview)
- [Mechanics of enumeration](#Mechanics-of-enumeration)
  - [Multiple latent variables](#Multiple-latent-variables)
  - [Examining discrete latent states](#Examining-discrete-latent-states)
  - [Indexing with enumerated variables](#Indexing-with-enumerated-variables)
- [Plates and enumeration](#Plates-and-enumeration)
  - [Dependencies among plates](#Dependencies-among-plates)
- [Time series example](#Time-series-example)
  - [How to enumerate more than 25 variables](#How-to-enumerate-more-than-25-variables)

```{code-cell} ipython3
import os
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide import AutoNormal
from pyro.ops.indexing import Vindex

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.7.0')
pyro.set_rng_seed(0)
```

## Overview <a class="anchor" id="Overview"></a>

Pyro's enumeration strategy ([Obermeyer et al. 2019](https://arxiv.org/abs/1902.03210)) encompasses popular algorithms including variable elimination, exact message passing, forward-filter-backward-sample, inside-out, Baum-Welch, and many other special-case algorithms. Aside from enumeration, Pyro implements a number of inference strategies including variational inference ([SVI](http://docs.pyro.ai/en/dev/inference_algos.html)) and monte carlo ([HMC](http://docs.pyro.ai/en/dev/mcmc.html#pyro.infer.mcmc.HMC) and [NUTS](http://docs.pyro.ai/en/dev/mcmc.html#pyro.infer.mcmc.NUTS)). Enumeration can be used either as a stand-alone strategy via [infer_discrete](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.discrete.infer_discrete), or as a component of other strategies. Thus enumeration allows Pyro to marginalize out discrete latent variables in HMC and SVI models, and to use variational enumeration of discrete variables in SVI guides.

+++

## Mechanics of enumeration  <a class="anchor" id="Mechanics-of-enumeration"></a>

The core idea of enumeration is to interpret discrete [pyro.sample](http://docs.pyro.ai/en/dev/primitives.html#pyro.sample) statements as full enumeration rather than random sampling. Other inference algorithms can then sum out the enumerated values. For example a sample statement might return a tensor of scalar shape under the standard "sample" interpretation (we'll illustrate with trivial model and guide):

```{code-cell} ipython3
def model():
    z = pyro.sample("z", dist.Categorical(torch.ones(5)))
    print(f"model z = {z}")

def guide():
    z = pyro.sample("z", dist.Categorical(torch.ones(5)))
    print(f"guide z = {z}")

elbo = Trace_ELBO()
elbo.loss(model, guide);
```

However under the enumeration interpretation, the same sample site will return a fully enumerated set of values, based on its distribution's [.enumerate_support()](https://pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution.enumerate_support) method.

```{code-cell} ipython3
elbo = TraceEnum_ELBO(max_plate_nesting=0)
elbo.loss(model, config_enumerate(guide, "parallel"));
```

Note that we've used "parallel" enumeration to enumerate along a new tensor dimension. This is cheap and allows Pyro to parallelize computation, but requires downstream program structure to avoid branching on the value of `z`. To support dynamic program structure, you can instead use "sequential" enumeration, which runs the entire model,guide pair once per sample value, but requires running the model multiple times.

```{code-cell} ipython3
elbo = TraceEnum_ELBO(max_plate_nesting=0)
elbo.loss(model, config_enumerate(guide, "sequential"));
```

Parallel enumeration is cheaper but more complex than sequential enumeration, so we'll focus the rest of this tutorial on the parallel variant. Note that both forms can be interleaved.

### Multiple latent variables <a class="anchor" id="Multiple-latent-variables"></a>

We just saw that a single discrete sample site can be enumerated via nonstandard interpretation. A model with a single discrete latent variable is a mixture model. Models with multiple discrete latent variables can be more complex, including HMMs, CRFs, DBNs, and other structured models. In models with multiple discrete latent variables, Pyro enumerates each variable in a different tensor dimension (counting from the right; see [Tensor Shapes Tutorial](http://pyro.ai/examples/tensor_shapes.html)). This allows Pyro to determine the dependency graph among variables and then perform cheap exact inference using variable elimination algorithms.

To understand enumeration dimension allocation, consider the following model, where here we collapse variables out of the model, rather than enumerate them in the guide.

```{code-cell} ipython3
@config_enumerate
def model():
    p = pyro.param("p", torch.randn(3, 3).exp(), constraint=constraints.simplex)
    x = pyro.sample("x", dist.Categorical(p[0]))
    y = pyro.sample("y", dist.Categorical(p[x]))
    z = pyro.sample("z", dist.Categorical(p[y]))
    print(f"  model x.shape = {x.shape}")
    print(f"  model y.shape = {y.shape}")
    print(f"  model z.shape = {z.shape}")
    return x, y, z
    
def guide():
    pass

pyro.clear_param_store()
print("Sampling:")
model()
print("Enumerated Inference:")
elbo = TraceEnum_ELBO(max_plate_nesting=0)
elbo.loss(model, guide);
```

### Examining discrete latent states <a class="anchor" id="Examining-discrete-latent-states"></a>

While enumeration in SVI allows fast learning of parameters like `p` above, it does not give access to predicted values of the discrete latent variables like `x,y,z` above. We can access these using a standalone [infer_discrete](http://docs.pyro.ai/en/dev/inference_algos.html#pyro.infer.discrete.infer_discrete) handler. In this case the guide was trivial, so we can simply wrap the model in `infer_discrete`. We need to pass a `first_available_dim` argument to tell `infer_discrete` which dimensions are available for enumeration; this is related to the `max_plate_nesting` arg of `TraceEnum_ELBO` via
```
first_available_dim = -1 - max_plate_nesting
```

```{code-cell} ipython3
serving_model = infer_discrete(model, first_available_dim=-1)
x, y, z = serving_model()  # takes the same args as model(), here no args
print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")
```

Notice that under the hood `infer_discrete` runs the model twice: first in forward-filter mode where sites are enumerated, then in replay-backward-sample model where sites are sampled. `infer_discrete` can also perform MAP inference by passing `temperature=0`. Note that while `infer_discrete` produces correct posterior samples, it does not currently produce correct logprobs, and should not be used in other gradient-based inference algorthms.

+++

### Indexing with enumerated variables

It can be tricky to use [advanced indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html) to select an element of a tensor using one or more enumerated variables. This is especially true in Pyro models where your model's indexing operations need to work in multiple interpretations: both sampling from the model (to generate data) and during enumerated inference. For example, suppose a plated random variable `z` depends on two different random variables:
```py
p = pyro.param("p", torch.randn(5, 4, 3, 2).exp(),
               constraint=constraints.simplex)
x = pyro.sample("x", dist.Categorical(torch.ones(4)))
y = pyro.sample("y", dist.Categorical(torch.ones(3)))
with pyro.plate("z_plate", 5):
    p_xy = p[..., x, y, :]  # Not compatible with enumeration!
    z = pyro.sample("z", dist.Categorical(p_xy)
```
Due to advanced indexing semantics, the expression `p[..., x, y, :]` will work correctly without enumeration, but is incorrect when `x` or `y` is enumerated. Pyro provides a simple way to index correctly, but first let's see how to correctly index using PyTorch's advanced indexing without Pyro:
```py
# Compatible with enumeration, but not recommended:
p_xy = p[torch.arange(5, device=p.device).reshape(5, 1),
         x.unsqueeze(-1),
         y.unsqueeze(-1),
         torch.arange(2, device=p.device)]
```
Pyro provides a helper [Vindex()[]](http://docs.pyro.ai/en/dev/ops.html#pyro.ops.indexing.Vindex) to use enumeration-compatible advanced indexing semantics rather than standard PyTorch/NumPy semantics. (Note the `Vindex` name and semantics follow the Numpy Enhancement Proposal [NEP 21](https://numpy.org/neps/nep-0021-advanced-indexing.html)). `Vindex()[]` makes the `.__getitem__()` operator broadcast like other familiar operators `+`, `*` etc. Using `Vindex()[]` we can write the same expression as if `x` and `y` were numbers (i.e. not enumerated):
```py
# Recommended syntax compatible with enumeration:
p_xy = Vindex(p)[..., x, y, :]
```
Here is a complete example:

```{code-cell} ipython3
@config_enumerate
def model():
    p = pyro.param("p", torch.randn(5, 4, 3, 2).exp(), constraint=constraints.simplex)
    x = pyro.sample("x", dist.Categorical(torch.ones(4)))
    y = pyro.sample("y", dist.Categorical(torch.ones(3)))
    with pyro.plate("z_plate", 5):
        p_xy = Vindex(p)[..., x, y, :]
        z = pyro.sample("z", dist.Categorical(p_xy))
    print(f"     p.shape = {p.shape}")
    print(f"     x.shape = {x.shape}")
    print(f"     y.shape = {y.shape}")
    print(f"  p_xy.shape = {p_xy.shape}")
    print(f"     z.shape = {z.shape}")
    return x, y, z
    
def guide():
    pass

pyro.clear_param_store()
print("Sampling:")
model()
print("Enumerated Inference:")
elbo = TraceEnum_ELBO(max_plate_nesting=1)
elbo.loss(model, guide);
```

When enumering within a plate (as described in the next section) ``Vindex`` can also be used together with capturing the plate index via ``with pyro.plate(...) as i`` to index into batch dimensions.  Here's an example with nontrivial event dimensions due to the ``Dirichlet`` distribution.

```{code-cell} ipython3
@config_enumerate
def model():
    data_plate = pyro.plate("data_plate", 6, dim=-1)
    feature_plate = pyro.plate("feature_plate", 5, dim=-2)
    component_plate = pyro.plate("component_plate", 4, dim=-1)
    with feature_plate: 
        with component_plate:
            p = pyro.sample("p", dist.Dirichlet(torch.ones(3)))
    with data_plate:
        c = pyro.sample("c", dist.Categorical(torch.ones(4)))
        with feature_plate as vdx:                # Capture plate index.
            pc = Vindex(p)[vdx[..., None], c, :]  # Reshape it and use in Vindex.
            x = pyro.sample("x", dist.Categorical(pc),
                            obs=torch.zeros(5, 6, dtype=torch.long))
    print(f"    p.shape = {p.shape}")
    print(f"    c.shape = {c.shape}")
    print(f"  vdx.shape = {vdx.shape}")
    print(f"    pc.shape = {pc.shape}")
    print(f"    x.shape = {x.shape}")

def guide():
    feature_plate = pyro.plate("feature_plate", 5, dim=-2)
    component_plate = pyro.plate("component_plate", 4, dim=-1)
    with feature_plate, component_plate:
        pyro.sample("p", dist.Dirichlet(torch.ones(3)))
    
pyro.clear_param_store()
print("Sampling:")
model()
print("Enumerated Inference:")
elbo = TraceEnum_ELBO(max_plate_nesting=2)
elbo.loss(model, guide);
```

## Plates and enumeration <a class="anchor" id="Plates-and-enumeration"></a>

Pyro [plates](http://docs.pyro.ai/en/dev/primitives.html#pyro.plate) express conditional independence among random variables. Pyro's enumeration strategy can take advantage of plates to reduce the high cost (exponential in the size of the plate) of enumerating a cartesian product down to a low cost (linear in the size of the plate) of enumerating conditionally independent random variables in lock-step. This is especially important for e.g. minibatched data.

To illustrate, consider a gaussian mixture model with shared variance and different mean.

```{code-cell} ipython3
@config_enumerate
def model(data, num_components=3):
    print(f"  Running model with {len(data)} data points")
    p = pyro.sample("p", dist.Dirichlet(0.5 * torch.ones(num_components)))
    scale = pyro.sample("scale", dist.LogNormal(0, num_components))
    with pyro.plate("components", num_components):
        loc = pyro.sample("loc", dist.Normal(0, 10))
    with pyro.plate("data", len(data)):
        x = pyro.sample("x", dist.Categorical(p))
        print("    x.shape = {}".format(x.shape))
        pyro.sample("obs", dist.Normal(loc[x], scale), obs=data)
        print("    dist.Normal(loc[x], scale).batch_shape = {}".format(
            dist.Normal(loc[x], scale).batch_shape))
        
guide = AutoNormal(poutine.block(model, hide=["x", "data"]))

data = torch.randn(10)
        
pyro.clear_param_store()
print("Sampling:")
model(data)
print("Enumerated Inference:")
elbo = TraceEnum_ELBO(max_plate_nesting=1)
elbo.loss(model, guide, data);
```

Observe that during inference the model is run twice, first by the `AutoNormal` to trace sample sites, and second by `elbo` to compute loss. In the first run, `x` has the standard interpretation of one sample per datum, hence shape `(10,)`. In the second run enumeration can use the same three values `(3,1)` for all data points, and relies on broadcasting for any dependent sample or observe sites that depend on data. For example, in the `pyro.sample("obs",...)` statement, the distribution has shape `(3,1)`, the data has shape`(10,)`, and the broadcasted log probability tensor has shape `(3,10)`.

For a more in-depth treatment of enumeration in mixture models, see the [Gaussian Mixture Model Tutorial](http://pyro.ai/examples/gmm.html) and the [HMM Example](http://pyro.ai/examples/hmm.html).

### Dependencies among plates <a class="anchor" id="Dependencies-among-plates"></a>

The computational savings of enumerating in vectorized plates comes with restrictions on the dependency structure of models (as described in ([Obermeyer et al. 2019](https://arxiv.org/abs/1902.03210))). These restrictions are in addition to the usual restrictions of conditional independence. The enumeration restrictions are checked by `TraceEnum_ELBO` and will result in an error if violated (however the usual conditional independence restriction cannot be generally verified by Pyro). For completeness we list all three restrictions:

#### Restriction 1: conditional independence
Variables within a plate may not depend on each other (along the plate dimension). This applies to any variable, whether or not it is enumerated. This applies to both sequential plates and vectorized plates. For example the following model is invalid:
```py
def invalid_model():
    x = 0
    for i in pyro.plate("invalid", 10):
        x = pyro.sample(f"x_{i}", dist.Normal(x, 1.))
```

#### Restriction 2: no downstream coupling
No variable outside of a vectorized plate can depend on an enumerated variable inside of that plate. This would violate Pyro's exponential speedup assumption. For example the following model is invalid:
```py
@config_enumerate
def invalid_model(data):
    with pyro.plate("plate", 10):  # <--- invalid vectorized plate
        x = pyro.sample("x", dist.Bernoulli(0.5))
    assert x.shape == (10,)
    pyro.sample("obs", dist.Normal(x.sum(), 1.), data)
```
 To work around this restriction, you can convert the vectorized plate to a sequential plate:
```py
@config_enumerate
def valid_model(data):
    x = []
    for i in pyro.plate("plate", 10):  # <--- valid sequential plate
        x.append(pyro.sample(f"x_{i}", dist.Bernoulli(0.5)))
    assert len(x) == 10
    pyro.sample("obs", dist.Normal(sum(x), 1.), data)
```

#### Restriction 3: single path leaving each plate
The final restriction is subtle, but is required to enable Pyro's exponential speedup

> For any enumerated variable `x`, the set of all enumerated variables on which `x` depends must be linearly orderable in their vectorized plate nesting.

This requirement only applies when there are at least two plates and at least three variables in different plate contexts. The simplest counterexample is a Boltzmann machine
```py
@config_enumerate
def invalid_model(data):
    plate_1 = pyro.plate("plate_1", 10, dim=-1)  # vectorized
    plate_2 = pyro.plate("plate_2", 10, dim=-2)  # vectorized
    with plate_1:
        x = pyro.sample("y", dist.Bernoulli(0.5))
    with plate_2:
        y = pyro.sample("x", dist.Bernoulli(0.5))
    with plate_1, plate2:
        z = pyro.sample("z", dist.Bernoulli((1. + x + y) / 4.))
        ...
```
Here we see that the variable `z` depends on variable `x` (which is in `plate_1` but not `plate_2`) and depends on variable `y` (which is in `plate_2` but not `plate_1`). This model is invalid because there is no way to linearly order `x` and `y` such that one's plate nesting is less than the other.

To work around this restriction, you can convert one of the plates to a sequential plate:
```py
@config_enumerate
def valid_model(data):
    plate_1 = pyro.plate("plate_1", 10, dim=-1)  # vectorized
    plate_2 = pyro.plate("plate_2", 10)          # sequential
    with plate_1:
        x = pyro.sample("y", dist.Bernoulli(0.5))
    for i in plate_2:
        y = pyro.sample(f"x_{i}", dist.Bernoulli(0.5))
        with plate_1:
            z = pyro.sample(f"z_{i}", dist.Bernoulli((1. + x + y) / 4.))
            ...
```
but beware that this increases the computational complexity, which may be exponential in the size of the sequential plate.

+++

## Time series example  <a class="anchor" id="Time-series-example"></a>

Consider a discrete HMM with latent states $x_t$ and observations $y_t$. Suppose we want to learn the transition and emission probabilities.

```{code-cell} ipython3
data_dim = 4
num_steps = 10
data = dist.Categorical(torch.ones(num_steps, data_dim)).sample()

def hmm_model(data, data_dim, hidden_dim=10):
    print(f"Running for {len(data)} time steps")
    # Sample global matrices wrt a Jeffreys prior.
    with pyro.plate("hidden_state", hidden_dim):
        transition = pyro.sample("transition", dist.Dirichlet(0.5 * torch.ones(hidden_dim)))
        emission = pyro.sample("emission", dist.Dirichlet(0.5 * torch.ones(data_dim)))

    x = 0  # initial state
    for t, y in enumerate(data):
        x = pyro.sample(f"x_{t}", dist.Categorical(transition[x]),
                        infer={"enumerate": "parallel"})
        pyro.sample(f"  y_{t}", dist.Categorical(emission[x]), obs=y)
        print(f"  x_{t}.shape = {x.shape}")
```

We can learn the global parameters using SVI with an autoguide.

```{code-cell} ipython3
hmm_guide = AutoNormal(poutine.block(hmm_model, expose=["transition", "emission"]))

pyro.clear_param_store()
elbo = TraceEnum_ELBO(max_plate_nesting=1)
elbo.loss(hmm_model, hmm_guide, data, data_dim=data_dim);
```

Notice that the model was run twice here: first it was run without enumeration by `AutoNormal`, so that the autoguide can record all sample sites; then second it is run by `TraceEnum_ELBO` with enumeration enabled. We see in the first run that samples have the standard interpretation, whereas in the second run samples have the enumeration interpretation.

For more complex examples, including minibatching and multiple plates, see the [HMM tutorial](https://github.com/pyro-ppl/pyro/blob/dev/examples/hmm.py).

+++

### How to enumerate more than 25 variables <a class="anchor" id="How-to-enumerate-more-than-25-variables"></a>

PyTorch tensors have a dimension limit of 25 in CUDA and 64 in CPU. By default Pyro enumerates each sample site in a new dimension. If you need more sample sites, you can annotate your model with  [pyro.markov](http://docs.pyro.ai/en/dev/poutine.html#pyro.poutine.markov) to tell Pyro when it is safe to recycle tensor dimensions. Let's see how that works with the HMM model from above. The only change we need is to annotate the for loop with `pyro.markov`, informing Pyro that the variables in each step of the loop depend only on variables outside of the loop and variables at this step and the previous step of the loop:
```diff
- for t, y in enumerate(data):
+ for t, y in pyro.markov(enumerate(data)):
```

```{code-cell} ipython3
def hmm_model(data, data_dim, hidden_dim=10):
    with pyro.plate("hidden_state", hidden_dim):
        transition = pyro.sample("transition", dist.Dirichlet(0.5 * torch.ones(hidden_dim)))
        emission = pyro.sample("emission", dist.Dirichlet(0.5 * torch.ones(data_dim)))

    x = 0  # initial state
    for t, y in pyro.markov(enumerate(data)):
        x = pyro.sample(f"x_{t}", dist.Categorical(transition[x]),
                        infer={"enumerate": "parallel"})
        pyro.sample(f"y_{t}", dist.Categorical(emission[x]), obs=y)
        print(f"x_{t}.shape = {x.shape}")

# We'll reuse the same guide and elbo.
elbo.loss(hmm_model, hmm_guide, data, data_dim=data_dim);
```

Notice that this model now only needs three tensor dimensions: one for the plate, one for even states, and one for odd states. For more complex examples, see the Dynamic Bayes Net model in the [HMM example](https://github.com/pyro-ppl/pyro/blob/dev/examples/hmm.py).

```{code-cell} ipython3

```
