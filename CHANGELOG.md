# Releases

## 0.10.1
### Added
* Improved SpMV performance

## 0.10.0
### Added
* CompiledFSM object whichstore the fsm and it's reversal.

## 0.9.1
### Fixed
* Do not allocate extra array while doing the backward step.

## 0.9.0
### Added
* All the different storage of FSM are now replaced with a single type:
  FSM which is internally a matrix-based FSM. Construction and
  inference algorithms are applied directly on this format.
* FSM constructor using a JSON-formatted string.
* Removed the old code `src/semirings` with semiring algebra and replace it
  with new external `Semirings` package.

## 0.8.0
* Added the `remove_label` function which replace the old `remove_eps`.
* Memory optimization of the forward-backward: the backward operates
  in-place avoiding to use an extra 3D tensor.
* This version is the one that was use for the 1st submission
  of the ICASSP 2022 conference.

## 0.7.0
* `pdfposteriors` supports for batch of sequence with varying lengths.
* refactorized the code in 3 sub-modules: `Semirings`, `FSMs` and `Inference`.
* the FSM API has several FSM implementation (`VectorFSM`,
  `HierarchicalFSM` and `MatrixFSM`)
* the FSM operations can be used are agnostic to the FSM-implementation
* added tests for the FSM API
* FSM states don't carry the pdf-id anymore and there is no notion
  of "epsilon-state"

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
