# SPDX-License-Identifier: MIT

push!(LOAD_PATH, "../src")

using Documenter
using MarkovModels

DocMeta.setdocmeta!(MarkovModels, :DocTestSetup,
                    :(using MarkovModels), recursive = true)

makedocs(
    sitename = "MarkovModels.jl",
    modules = [MarkovModels],
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Semirings" => "semirings.md",
            "Finite State Machines" => "fsm.md",
            "Inference" => "inference.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/lucasondel/MarkovModels.jl.git",
)
