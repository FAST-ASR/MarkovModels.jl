# Installation

## Installation of Julia

The MarkovModels package was developped and tested with Julia 1.5.0.
If you haven't installed Julia already, follow the instruction
[here](https://julialang.org/downloads/).

!!! tip
    It is a common practice in Julia to use non-ascii characters while
    coding such as greek letters or mathematical symbols. We highly
    recommend to add Julia support to your editor to easily access
    these special characters. Plugin for [vim](https://www.vim.org/)/[neovim](https://neovim.io/)
    and [emacs](https://www.gnu.org/software/emacs/) can be found
    [here](https://github.com/JuliaEditorSupport).

## Installation of MarkovModels

In the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)
prompt, press `]` to enther the Pkg REPL and then type:
```
(@v1.5) pkg> add https://github.com/BUTSpeechFIT/MarkovModels
```

This will install the package and its dependencies into your Julia
installation.

