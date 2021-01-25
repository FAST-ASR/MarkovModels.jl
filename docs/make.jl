using Documenter

push!(LOAD_PATH, "../")
using MarkovModels

makedocs(
    sitename = "MarkovModels",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Finite State Machines" => "fsm.md",
            "Inference" => "inference.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/BUTSpeechFIT/MarkovModels.git",
)
