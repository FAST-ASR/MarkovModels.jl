# Releases

## 0.6.1
* fixed typo `arc!` instead of `addarc!`

## 0.6.0
* added `maxstateposteriors` and `bestpath` algorithms
* can make union of compiled FSMs on GPU.

## 0.5.0

* added batch computation of the forward-backward algorithm
* added `remove_eps` function to remove emitting states
* CompiledFSM structure stores the state -> pdf mapping (and pdf ->
  state mapping) as two sparse matrices to deal with
  the case when the state id is different from the pdf index
* added `union` function to group several fsms together
* fixed determinize never end for FSM with cycles (issue #16)
* fixed duplicate initial state when determinizing

## 0.4.0

* added best path decoding
* improved user api by "hiding" the conversion of the state llhs to
  the appropriate semi-field
* improved benchmark

## 0.3.0

* added batch version of the forward-backward algorithm
* added GPU support for the forward-backward algorithm
* remove pruning option for the forward-backward algorithm

## 0.2.0

* refactoring of the FSM API.
* added a `compile` function to convert an inference graph into a
  compact matrix-based format suitable for fast inference
* added support for dense/sparse CPU/GPU version of the
  forward-backward algorithm
  * the forward-backward implementation does not rely on the graph
    representation anymore
* added unit tests
* added `LogSemifield` type used by the graph and inference API

## 0.1.0

* initial release
