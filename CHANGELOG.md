# Releases

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
