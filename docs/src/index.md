# MarkovModels Documentation

[MarkovModels](https://github.com/BUTSpeechFIT/MarkovModels) is a
[Julia](https://julialang.org/) package to use (Hidden) Markov Models
for probabilistic inference.

See the project on [github](https://github.com/BUTSpeechFIT/MarkovModels).

To quickly get started, have a look at our examples:
* [Building a simple ASR decoder](https://github.com/BUTSpeechFIT/MarkovModels/blob/master/examples/demo.ipynb)
* [Baum-Welch (forward-backward) algorithm](https://github.com/BUTSpeechFIT/MarkovModels/blob/master/examples/inference.ipynb)

## Authors

* [Lucas Ondel](https://lucasondel.github.io), Brno University of Technology
* Martin Kocour, Brno University of Technology

## Installation

The package can be installed with the Julia package manager. From the
Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia
pkg> add MarkovModels
```

## Manual Outline

```@contents
Pages = ["fsm.md", "inference.md"]
```

