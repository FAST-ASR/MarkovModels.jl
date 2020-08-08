using Documenter

push!(LOAD_PATH, "../")
using MarkovModels

makedocs(
    sitename = "MarkovModels",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
)

deploydocs(
    repo = "github.com/BUTSpeechFIT/MarkovModels.git",
)
