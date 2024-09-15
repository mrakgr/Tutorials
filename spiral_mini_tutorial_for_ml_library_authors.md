<!-- TOC -->

- [Intro](#intro)
- [Printing in Spiral](#printing-in-spiral)
- [Tensors in Spiral](#tensors-in-spiral)
- [Spiral's staging capabilities](#spirals-staging-capabilities)
- [Partially applied functions can be passed through language boundaries in Spiral](#partially-applied-functions-can-be-passed-through-language-boundaries-in-spiral)
- [A brief look at Spiral's recursive unions](#a-brief-look-at-spirals-recursive-unions)
- [Maps on the GPU in Spiral](#maps-on-the-gpu-in-spiral)
- [The softmax is easy to implement in Spiral](#the-softmax-is-easy-to-implement-in-spiral)
- [Outro](#outro)

<!-- /TOC -->

# Intro

Not a bug or a question. Since you're working on an ML library, I just wanted to showcase some of my own work.

I have a functional PL called [Spiral](https://github.com/mrakgr/The-Spiral-Language). I started work on it back in late 2016 for the explicit purpose of making ML libraries and having it easily be adaptible to novel ML hardware. You can see me programming in it on [my Youtube channel](https://www.youtube.com/playlist?list=PL04PGV4cTuIVP50-B_1scXUUMn8qEBbSs). It even has an [ML library](https://github.com/mrakgr/Spiral-s-ML-Library). What I've implemented in that repo (besides the ML library) is are Leduc and HU NL Texas Hold'em implementations. They have a web UI, so you can run the games directly on the GPU while using the ML library to run the agents. Everything on the GPU compiles into a single fused kernel.

I've also created serialization libraries so that arbitrary (non-recursive) union types can easily be serialized from the Python side onto the Cuda side and vice versa.

# Printing in Spiral

Let me give a little demo, to showcase the language strengths. Here is how printing is done. Note that the examples look prettier in the editor which has type hovers and syntax highlighting.

```spiral
open corebase
open corecuda
open coreext

open tensorm

inl main() =
    inl t : tensor (int * int) float = cupy.random_normal {mean=0; std=1} (5,5)
    console.write_ln "Printing this on host."
    console.write_ln t
    run fun () =>
        if rangem.threads_in_grid().from = 0 then
            console.write_ln "Printing this on the device."
            console.write_ln t
```

When I compile [this](https://github.com/mrakgr/Spiral-s-ML-Library/blob/1ee9d15bd2e44be7a2695d15d2332811a754ffd5/tests/showcase1.spi) and run [it](https://github.com/mrakgr/Spiral-s-ML-Library/blob/1ee9d15bd2e44be7a2695d15d2332811a754ffd5/tests/showcase1.py#L213) here is what I get.

```
Printing this on host.
[[0.053561; 0.341148; -0.505733; 0.203342; 1.096532]; [0.048035; 0.363875; -0.654670; -0.063081; 0.098163]; [-1.404943; -0.144419; 0.434076; -0.549689; -1.521033]; [0.207757; -1.594985; -0.171212; -0.920335; -1.507881]; [-1.784490; -0.932659; -1.700014; -0.100949; 0.377217]]
Printing this on the device.
[[0.053561; 0.341148; -0.505733; 0.203342; 1.096532]; [0.048035; 0.363875; -0.654670; -0.063081; 0.098163]; [-1.404943; -0.144419; 0.434076; -0.549689; -1.521033]; [0.207757; -1.594985; -0.171212; -0.920335; -1.507881]; [-1.784490; -0.932659; -1.700014; -0.100949; 0.377217]]
```

The `console.write_ln` uses a global semaphore to make sure that only a single thread is doing printing at any given time, so it's easy to print out complex datatypes without their outputs getting mixed up as would happen when you chain print statements in regular Cuda code.

Here is another [example](https://github.com/mrakgr/Spiral-s-ML-Library/blob/70c664dcce47dad679849204f8bf6263268179f9/tests/showcase2.spi).

```spiral
open corebase
open corecuda
open coreext

open tensorm

inl main() =
    inl t : tensor (int * int) float = cupy.random_normal {mean=0; std=1} (5,5)
    console.write_ln "Printing this on host."
    console.write_ln t
    run fun () =>
        inl tid = rangem.threads_in_grid().from
        if tid < 4 then
            console.write_ln {tensor=t; tid}
```

When I run [this script](https://github.com/mrakgr/Spiral-s-ML-Library/blob/70c664dcce47dad679849204f8bf6263268179f9/tests/showcase2.py), I get...

```
Printing this on host.
[[-0.409157; 1.515481; 1.616163; 0.543908; 1.375615]; [1.484471; 0.322833; 1.882504; 0.043646; -0.468472]; [0.733591; 0.972111; -0.093996; -0.987146; -0.383688]; [0.093093; -0.491220; 0.444016; -0.560113; 1.083763]; [-0.866570; 1.273009; -0.109076; 0.786436; 1.214065]]
{tensor = [[-0.409157; 1.515481; 1.616163; 0.543908; 1.375615]; [1.484471; 0.322833; 1.882504; 0.043646; -0.468472]; [0.733591; 0.972111; -0.093996; -0.987146; -0.383688]; [0.093093; -0.491220; 0.444016; -0.560113; 1.083763]; [-0.866570; 1.273009; -0.109076; 0.786436; 1.214065]]; tid = 0}
{tensor = [[-0.409157; 1.515481; 1.616163; 0.543908; 1.375615]; [1.484471; 0.322833; 1.882504; 0.043646; -0.468472]; [0.733591; 0.972111; -0.093996; -0.987146; -0.383688]; [0.093093; -0.491220; 0.444016; -0.560113; 1.083763]; [-0.866570; 1.273009; -0.109076; 0.786436; 1.214065]]; tid = 1}
{tensor = [[-0.409157; 1.515481; 1.616163; 0.543908; 1.375615]; [1.484471; 0.322833; 1.882504; 0.043646; -0.468472]; [0.733591; 0.972111; -0.093996; -0.987146; -0.383688]; [0.093093; -0.491220; 0.444016; -0.560113; 1.083763]; [-0.866570; 1.273009; -0.109076; 0.786436; 1.214065]]; tid = 2}
{tensor = [[-0.409157; 1.515481; 1.616163; 0.543908; 1.375615]; [1.484471; 0.322833; 1.882504; 0.043646; -0.468472]; [0.733591; 0.972111; -0.093996; -0.987146; -0.383688]; [0.093093; -0.491220; 0.444016; -0.560113; 1.083763]; [-0.866570; 1.273009; -0.109076; 0.786436; 1.214065]]; tid = 3}
```

# Tensors in Spiral

Tensors are very important building blocks for an ML library, so a decent amount of effort went into designing a language that could support their proper implementation. They can have arbitrary number of dimensions, but more interestingly, internally they are laid out as a structure-of-arrays.

Here is an [example](https://github.com/mrakgr/Spiral-s-ML-Library/blob/fe9aaf7b89ebb6bd66fb1ce4deb6878c7c11e019/tests/showcase3.spi).

```
open corebase
open corecuda
open coreext

open tensorm

inl main() =
    inl t : tensor (int * int * int) {x : int; y : float; z : bool} = tensor_create (2,3,4)
    run fun () =>
        inl q = tensor_index (0,0,0) t
        ()
```

If you take a look at how the kernel is compiled on the Cuda side, [you'll see](https://github.com/mrakgr/Spiral-s-ML-Library/blob/fe9aaf7b89ebb6bd66fb1ce4deb6878c7c11e019/tests/showcase3.py#L205):

```cpp
extern "C" __global__ void entry0(int * v0, float * v1, bool * v2) {
    int v3;
    v3 = v0[0l];
    float v4;
    v4 = v1[0l];
    bool v5;
    v5 = v2[0l];
    return ;
}
```

And here is how they are laid out on the [Python side](https://github.com/mrakgr/Spiral-s-ML-Library/blob/fe9aaf7b89ebb6bd66fb1ce4deb6878c7c11e019/tests/showcase3.py#L276).

```python
def main_body():
    v0 = cp.empty(24,dtype=cp.int32)
    v1 = cp.empty(24,dtype=cp.float32)
    v2 = cp.empty(24,dtype=cp.bool_)
```

They get tracked as separate arrays and passed into the kernel as pointers. This functionality makes it easy to pass compound data types from the Python side onto the Cuda side.

# Spiral's staging capabilities

In the previous example, most of the data making up the tensor has been passed at compile time. By that I mean, the tensor dimensions, strides and offsets. By using `dyn` we can push the tensor dimensions to runtime, forcing that data to be tracked during runtime.

Here is [an example](https://github.com/mrakgr/Spiral-s-ML-Library/blob/1967d92e621ce91a9b0ad9bde88e05b35457958f/tests/showcase4.spi).

```spiral
open corebase
open corecuda
open coreext

open tensorm

inl main() =
    inl t : tensor (int * int * int) {x : int; y : float; z : bool} = tensor_create (dyn (2,3,4))
    run fun () =>
        inl q = tensor_index (0,0,0) t
        ()
```

If you take a look at how it gets compiled now, you'll see a completely [different result](https://github.com/mrakgr/Spiral-s-ML-Library/blob/1967d92e621ce91a9b0ad9bde88e05b35457958f/tests/showcase4.py#L205).

```cpp
extern "C" __global__ void entry0(int * v0, int v1, int v2, float * v3, bool * v4, int v5, int v6) {
    assert("Tensor range check" && 0 <= 0l && 0l < v5);
    assert("Tensor range check" && 0 <= 0l && 0l < v6);
    assert("Tensor range check" && 0 <= 0l && 0l < v2);
    int v7;
    v7 = v0[0l];
    float v8;
    v8 = v3[0l];
    bool v9;
    v9 = v4[0l];
    return ;
}
```

You can see the 3 dimensions and 1 of the strides getting passed at runtime. Furthermore, now that the dimensions are known only at compile time, the asserts are generated to make sure that the tensor indexing is not out of bounds.

I myself thought that there'd be more variables getting passed at runtime. We can force it to partially evaluate the entire tensor by doing the [following](https://github.com/mrakgr/Spiral-s-ML-Library/blob/cutlass_intro/tests/showcase5.spi).

```spiral
open corebase
open corecuda
open coreext

open tensorm

inl main() =
    inl t : tensor (int * int * int) {x : int; y : float; z : bool} = dyn (tensor_create (2,3,4))
    run fun () =>
        inl q = tensor_index (0,0,0) t
        ()
```

Now it has to pass in all the [tensor data explicitly](https://github.com/mrakgr/Spiral-s-ML-Library/blob/bd5b3a5faad10b35b2b3655c94313628ff330e50/tests/showcase5.py#L205).

```
extern "C" __global__ void entry0(int * v0, int v1, int v2, int v3, int v4, float * v5, int v6, bool * v7, int v8, int v9, int v10, int v11) {
    assert("Tensor range check" && 0 <= 0l && 0l < v9);
    assert("Tensor range check" && 0 <= 0l && 0l < v10);
    assert("Tensor range check" && 0 <= 0l && 0l < v11);
    int v12;
    v12 = v0[v1];
    float v13;
    v13 = v5[v6];
    bool v14;
    v14 = v7[v8];
    return ;
}
```

# Partially applied functions can be passed through language boundaries in Spiral

These kinds of partial evaluation capabilities, where certain value can be staged at either the compile or runtime are quite useful. For one, functions can also be evaluated at compile time, and what that means is that it becomes possible to freely exchange lambdas between the Python and the Cuda side.

[Behold.](https://github.com/mrakgr/Spiral-s-ML-Library/blob/5c3ddd19a46fe802c393f7edc07b0bc09b8dd6ad/tests/showcase6.spi)

```spiral
open corebase
open corecuda
open coreext

open tensorm

let add a b = a + b
inl main() =
    inl partially_applied_add = add 1i32
    inl _ = partially_applied_add 2
    run fun () =>
        inl _ = partially_applied_add 3
        ()
```

Here is how the generated code looks like on the [Cuda side](https://github.com/mrakgr/Spiral-s-ML-Library/blob/5c3ddd19a46fe802c393f7edc07b0bc09b8dd6ad/tests/showcase6.py#L205).

```cpp
__device__ int add_0(int v0, int v1){
    int v2;
    v2 = v0 + v1;
    return v2;
}
extern "C" __global__ void entry0(int v0) {
    int v1;
    v1 = 3l;
    int v2;
    v2 = add_0(v0, v1);
    return ;
```

An integer gets passed into the kernel and then it gets passed along the `3` into the `add` function. On the Python side, it will also get [generated](https://github.com/mrakgr/Spiral-s-ML-Library/blob/5c3ddd19a46fe802c393f7edc07b0bc09b8dd6ad/tests/showcase6.py#L280).

```py
def method0(v0 : i32, v1 : i32) -> i32:
    v2 = v0 + v1
    del v0, v1
    return v2
def main_body():
    v0 = 1
    v1 = 2
    v2 = method0(v0, v1)
```

Why is all this so useful?

# A brief look at Spiral's recursive unions

As mentioned, Spiral has an [ML library](https://github.com/mrakgr/Spiral-s-ML-Library/blob/5c3ddd19a46fe802c393f7edc07b0bc09b8dd6ad/ml/layers.spi#L19). You can see how its graph looks like.

```spiral
union rec graph t =
    | BlockMap :: forall dim t. 
        (exists a. (layer_state -> a -> t) 
            * graph (tensor (int * dim) a) 
            * option (graph (tensor (int * dim) t))) 
        -> graph (tensor (int * dim) t)
    | BlockRowMap :: forall dim t.
        (exists a. (layer_state -> primitives.row_config -> tensor (int * int) a -> dim -> tensor (int * int) int -> tensor (int * int) t) 
            * graph (tensor (int * dim * int) a)
            * option (graph (tensor (int * dim * int) t)))
        -> graph (tensor (int * dim * int) t)
    | BlockRowReduce :: forall dim t.
        (exists a. (layer_state -> primitives.row_config -> tensor (int * int) a -> dim -> tensor (int * int) int -> t) 
            * graph (tensor (int * dim * int) a)
            * option (graph (tensor (int * dim) t)))
        -> graph (tensor (int * dim) t)
```

I am not going to replicate it in full in the above code segment, but what Spiral's staging capabilities offer us is the power to do various compilation passes on a graph defined on the host side and then transfer it witout any compilation footprint to the Cuda side. You can calculate all the sizes, allocate the graph in bulk as one array on the host, and then derive the input and output tensors for a particular node on the Cuda side. For large graph that could mean compressing 100s of potential pointers into a single one. Or two, because in Spiral's ML library the parameter and IO spaces are separate.

Much like functions, union types that are known at compile time can be passed freely between the Python and Cuda backends even if they are recursive.

# Maps on the GPU in Spiral

Unlike some other languages, Spiral's tensors aren't built into the language, but implemented as a [part of a library](https://github.com/mrakgr/Spiral-s-ML-Library/blob/ae37894e8c5e6e2c8d73ba191ae0dad273366600/corebase/tensorm/tensor_main.spi) instead. They have various functionality for splitting, taking views, apply certain dimensions and so on. Using them, it's easy to implement a [map operation](https://github.com/mrakgr/Spiral-s-ML-Library/blob/5c3ddd19a46fe802c393f7edc07b0bc09b8dd6ad/ml/primitives.spi#L12) in Cuda.

```spiral
// Maps all the elements of a tensor given the mapping function.
inl map_ forall dim a b. loop' (f : a -> b) (from : tensor dim a) (to : tensor dim b) : () = 
    assert (from.dim = to.dim) "The dimensions of the two inputs to the map kernel need to be the same."
    inl from,to = factorize_sizeof_16 from, factorize_sizeof_16 to
    loop' (fst from.dim) fun i => 
        inl from, to = apply i from, apply i to
        inl l_from, l_to = tensor_create from.dim, tensor_create to.dim
        memcpy_sync (l_from, from)
        pragma.unroll fun _ =>
            loop.linear from.dim fun j => 
                tensor_set j (tensor_index j l_from |> f) l_to
        memcpy_sync (to, l_to)
```

This map is a bit more complex than you'd expect a regular map to be due to the 16-byte factorization. The innermost (rightmost) dimension of a tensor is split into 16-byte chunks to allow the 128-bit global load instructions to be done. Then a map is done on tensors in local memory before being send into the output tensor. It's not too complex, but it is complex enough that you wouldn't want to be writing it out every time, which is why it's been factored out into a function. Here is [an example](https://github.com/mrakgr/Spiral-s-ML-Library/blob/87f166c3770e7345f3c3db5786dfda91dce85627/tests/showcase7.spi) of it in use.

It has blockwise `map` and gridwise `grid_map` variants, the following one you'd want to use when you're debugging with only a single block launched, or when the input and output tensors differ between the blocks.

```spiral
open corebase
open corecuda
open coreext

open tensorm

inl main() =
    inl in_ : tensor (int * int) float = cupy.zeros (2,8)
    inl out = tensor_create in_.dim
    run fun () =>
        ml.primitives.map (fun x => x + 5) in_ out
    console.write_ln {in_ out}
```

When I run [the script](https://github.com/mrakgr/Spiral-s-ML-Library/blob/87f166c3770e7345f3c3db5786dfda91dce85627/tests/showcase7.py), here is what I get in the terminal.

```
{in_ = [[0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000]; [0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000]; [0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000]; [0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000; 0.000000]]; out = [[5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000]; [5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000]; [5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000]; [5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000; 5.000000]]} 
```

# The softmax is easy to implement in Spiral

The map is the simplest loop in the `primitives` module, so I've used it for illustration purposes. There are more advanced loops.

In particular `row_map` and `row_reduce` allow us to load the tensors into local memory and operate them using `local_map`, `local_scan` and `local_reduce` operations. These two functions allow us to implement the majority of machine learning functions in a fused manner.

For example...

```spiral
// Numerically stable softmax.
inl local_softmax config x =
    inl average = local_average config x
    inl x = local_map (fun x => exp (x - average)) x
    inl sum = local_sum config x
    local_map (fun x => x / sum) x
```

The above is how the [softmax](https://github.com/mrakgr/Spiral-s-ML-Library/blob/87f166c3770e7345f3c3db5786dfda91dce85627/ml/primitives.spi#L527) is implemented in the library using just those few local primitives.

The following is [an example](https://github.com/mrakgr/Spiral-s-ML-Library/blob/db6f5a61a3a778b534a31409ae7e7721833bef7a/tests/showcase8.spi) of it in use.

```spiral
open corebase
open corecuda
open coreext

open tensorm

inl main() =
    inl in_ : tensor (int * int) float = cupy.random_normal {std=1; mean=0} (32,4) // row map has minimum size restrictions
    inl out = tensor_create in_.dim
    run fun () =>
        open ml.primitives
        row_map (fun config x _ _ => local_softmax config x) in_ out
    console.write_ln (view (fun a,b => {from=0,0; nearTo=4,b}) out) // print only the first 4 rows
```

Once I run [the script](https://github.com/mrakgr/Spiral-s-ML-Library/blob/db6f5a61a3a778b534a31409ae7e7721833bef7a/tests/showcase8.py), here is what I see in the terminal.

```
[[0.321847; 0.228128; 0.161914; 0.288111]; [0.084648; 0.164873; 0.448229; 0.302250]; [0.053764; 0.122373; 0.234389; 0.589474]; [0.105210; 0.283261; 0.217264; 0.394266]]
```

Using this function, it really easy to implement almost all machine learning operations that exist.

# Outro

This should be enough for you to get the picture of what Spiral is capable of so I'll stop the tutorial here.

I am interested in RL, and not supervised learning, so even though the library has a working parallel [tabular CFR](https://github.com/mrakgr/Spiral-s-ML-Library/blob/db6f5a61a3a778b534a31409ae7e7721833bef7a/ml/cfr.spi#L191) implementation, it doesn't have backpropagation. It could be added without too much hassle if there is any interest in the language from the people here.

The language compiles to Python and Cuda, and it has other backends besides that. More could be added in the future, as it was designed to be flexible for novel kinds of hardware. To be honest, I am disappointed that in 2024 I've gone back to programming Nvidia GPU instead of novels kinds of hardware the language was envisioned for, but AI chip startups have been a massive disappointment to me.

If you have any questions, leave them here or open an issue on the Spiral repos. If you don't want to use the language, that is fine, I just wanted you to know that it exists and to give you a rough overview of its strengths. Maybe you'll find something you like.

If you are interested you can grab it from VS Code marketplace. Just make sure to install the .NET 8 SDK first. The docs are on the repo.

On my own end, I am interested in what you'd like to see. More than Spiral, if you want somebody with PL skills and are interested in sending me paid work, don't hesitate to get in touch.