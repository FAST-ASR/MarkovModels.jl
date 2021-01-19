# Finite State Machines

The MarkovModels package represents Markov chains as probabilistic a
Finite State Machine (FSM).  Here is an example of FSM as used by the
package:

![alternative text](images/examplefsm.svg)

The double edge circle node with the label "<s>" (respectively "</s>")
is the initial (respectively final) state of the FSM. States with light
blue background color are *emitting states*, that is, they are
associated with a probability density function index (`pdfindex`). If
they have no label, this index is use when displaying the node - as in
the example above. White circle node with a label written inside are
*non-emitting labeled states*. The states represented as point are
neither emitting nor have a label. Finally, the number on the links
are the log-probabilities to move from one state to another.

!!! note
    To be able to visualize FSMs as in the example above when using
    [IJulia](https://github.com/JuliaLang/IJulia.jl), make sure that
    the `dot` program (from [graphviz](https://graphviz.org/)) is
    available in your shell `PATH` variable. Also, you won't be able
    to visualize the FSM in the REPL.

In the following, we present the tools provided by the package
manipulate such FSM. All the examples below assume that you
have already imported the MarkovModels package by doing `using
MarkovModels`.

## Creating FSMs

FSMs are represented by the following structure:
```@docs
FSM
```

To create an FSM object simply type:
```julia
fsm = FSM{Float64}()
```
![alternative text](images/initfsm.svg)

When created, the FSM has only two states: the initial state and the
final state. FSMs cannot have multiple initial for final states.

You can add states to the FSM by using the function `addstate!`:
```julia
s1 = addstate!(fsm, pdfindex = 1)
s2 = addstate!(fsm, pdfindex = 2, label = "a")
s3 = addstate!(fsm, label = "b")
s4 = addstate!(fsm)
```
![alternative text](images/addstate.svg)

Note that a state can be:
  * emitting and labeled
  * emitting only
  * labeled only
  * non-emitting and non-labeled (nil state)
The initial and final states are specific nil states.

The `link!` allows to add weighted arcs between states:
```julia
link!(initstate(fsm), s1)
link!( s1, s1, log(1/2))
link!(s1, s2, log(1/2))
link!(s2, s3)
link!(s3, s4)
link!(s4, finalstate(fsm))
```
![alternative text](images/links.svg)

Finally, we provide a special constructor for convenience:
```@docs
LinearFSM
```

For instance,
```julia
fsm = LinearFSM(Float32, ["a", "b", "c"], Dict("a" => 1, "b" => 2, "c" => 3))
```
![](images/linearfsm.svg)

## FSM operations

```@meta
CurrentModule = MarkovModels
```

### Composition

```@docs
compose
```

#### Example

```julia
fsm = union(LinearFSM(["a", "b"]), LinearFSM(["c"])) |> weightnormalize
subfsms = subfsms = Dict(
    "a" => LinearFSM(["a1", "a2", "a3"], Dict("a1"=>1, "a2"=>2, "a3"=>3)),
    "b" => LinearFSM(["b1", "b2"], Dict("b1"=>4, "b2"=>5)),
    "c" => LinearFSM(["c1", "c2"], Dict("c1"=>6, "c2"=>1))
)
compose(subfsms, fsm)
```

Input :
  * `fsm`
  ![](images/compose_input1.svg)
  * `subfsms["a"]`
  ![](images/compose_input2.svg)
  * `subfsms["b"]`
  ![](images/compose_input3.svg)
  * `subfsms["c"]`
  ![](images/compose_input4.svg)
Output:
  ![](images/compose_output.svg)

Alternatively, FSMs can be composed with the `∘` operator:
```julia
fsm ∘ sufsms
```

### Concatenation

```@docs
concat
```

#### Example

```julia
fsm1 = LinearFSM(["a", "b"])
fsm2 = LinearFSM(["c", "d"])
fsm3 = LinearFSM(["e"])
concat(fsm1, fsm2, fsm3)
```
Input:
  * `fsm1`
  ![](images/concat_input1.svg)
  * `fsm2`
  ![](images/concat_input2.svg)
  * `fsm3`
  ![](images/concat_input3.svg)

Output:
  ![](images/concat_output.svg)

### Determinization

```@docs
determinize
```

### Example

```julia
fsm = FSM{Float64}()
s1 = addstate!(fsm, label = "a")
s2 = addstate!(fsm, label = "b", pdfindex = 1)
link!(s1, s2, log(1/2))
link!(s1, s2, log(1/2))
link!(initstate(fsm), s1)
link!(s2, finalstate(fsm))
fsm |> determinize
```

Input:
![](images/determinize_input.svg)

Output:
![](images/determinize_output.svg)


### Minimization

```@docs
minimize
```
#### Example
```
fsm = union(LinearFSM(["a", "b", "c"], Dict("a"=>1)), LinearFSM(["a", "d", "c"], Dict("a"=>1)))
fsm |> minimize
```

Input:

![](images/minimize_input.svg)

Output:

![](images/minimize_output.svg)

### Nil states removal
```@docs
removenilstates
```

#### Example

```julia
fsm = LinearFSM(["a", "b"], Dict("a" => 1))
nil = addstate!(fsm)
link!(initstate(fsm), nil)
link!(nil, finalstate(fsm))
fsm = fsm |> weightnormalize
fsm |> removenilstates
```
Input:

![](images/rmnil_input.svg)

Ouput:

![](images/rmnil_output.svg)

### Transposition

```@docs
Base.transpose(::AbstractFSM{T}) where T
```
#### Example
```julia
fsm = LinearFSM(["a", "b", "c"])
transpose(fsm)
```
Input:

![](images/transpose_input.svg)

Output:

![](images/transpose_output.svg)

### Union

```@docs
Base.union(::AbstractFSM{T}, ::AbstractFSM{T}) where T
```

#### Example

```julia
fsm1 = LinearFSM(["a", "b", "c"], Dict("a"=>1))
fsm2 = LinearFSM(["a", "d", "c"], Dict("a"=>1))
union(fsm1, fsm2)
```
Input:

  * `fsm1`

    ![See the online documentation to visualize the image](images/union_input1.svg)
  * `fsm2`

    ![See the online documentation to visualize the image](images/union_input2.svg)

Output:

![](images/union_output.svg)


### Weight normalization

```@docs
weightnormalize
```

#### Example

```julia
fsm = union(LinearFSM(["a", "b"]), LinearFSM(["c", "d"]))
for s in states(fsm)
    if ! isinit(s) && ! isfinal(s)
        link!(s, s)
    end
end
fsm |> weightnormalize
```
Input:

![](images/wnorm_input.svg)

Output:

![](images/wnorm_output.svg)

