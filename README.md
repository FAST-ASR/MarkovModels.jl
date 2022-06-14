# MarkovModels

*Julia package for inference with (Hidden) Markov Models on CPU or
GPU.*

| **Documentation**  |
|:------------------:|
|[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://FAST-ASR.github.io/MarkovModels.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://FAST-ASR.github.io/MarkovModels.jl/dev)|

Here is a small [benchmark](https://github.com/lucasondel/MarkovModels.jl/tree/master/misc/benchmark) to see how our implementation performs.

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type ] to enter the Pkg REPL mode and run:

```julia
pkg> add MarkovModels
```

Have a look at the [documentation](https://lucasondel.github.io/MarkovModels.jl/stable)
or the [examples](https://github.com/lucasondel/MarkovModels.jl/tree/master/examples) 
to get started.

## Reference 

The formal description of the implementation can be found [here](https://hal.archives-ouvertes.fr/hal-03434552/document).  If you use this package, we kindly ask you to cite our work: 
```
@unpublished{ondel:hal-03434552,
  TITLE = {{GPU-Accelerated Forward-Backward Algorithm with Application to Lattic-Free MMI}},
  AUTHOR = {Ondel, Lucas and Lam-Yee-Mui, L{\'e}a-Marie  and Kocour, Martin and Filippo, Caio and Luk{\'a}s Burget, Corro},
  URL = {https://hal.archives-ouvertes.fr/hal-03434552},
  NOTE = {working paper or preprint},
  YEAR = {2021},
  MONTH = Nov,
  KEYWORDS = {Lattice-Free MMI ; end-to-end ASR ; Julia language ; forward-backward},
  PDF = {https://hal.archives-ouvertes.fr/hal-03434552/file/paper.pdf},
  HAL_ID = {hal-03434552},
  HAL_VERSION = {v1},
}
```

## Authors

* [Lucas Ondel](https://lucasondel.github.io/)   
* Martin Kocour    
