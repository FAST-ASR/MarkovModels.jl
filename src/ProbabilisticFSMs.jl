
module ProbabilisticFSMs

export StateID
export initstateid
export finalstateid

export Link

export State
export isemitting


export FSM
export LinearFSM
export addstate!
export link!

export initstate
export finalstate
export name

export backward
export children
export emittingstates
export forward
export links
export parents
export states

include("fsm.jl")

# Pretty display of the graph in IJulia.
function Base.show(io, ::MIME"image/svg+xml", fsm::FSM)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    for state in states(fsm)
        shape = isemitting(state) ? "circle" : "point"
        label = "$(state.id):$(name(fsm, state))"
        write(dotfile, "$(state.id) [ shape=\"$(shape)\" label=\"$label \"];\n")
    end

    for link in links(fsm)
        weight = round(link.weight, digits = 3)
        write(dotfile, "$(link.src.id) -> $(link.dest.id) [ label=\"$(weight)\" ];\n")
    end
    write(dotfile, "}\n")
    close(dotfile)
    run(`dot -Tsvg $(dotpath) -o $(svgpath)`)

    xml = read(svgfile, String)
    write(io, xml)

    close(svgfile)

    rm(dotpath)
    rm(svgpath)
end

#######################################################################
# Concrete graph implementation

export PruningStrategy
export ThresholdPruning
export nopruning

#######################################################################
# Pretty display the sparse matrix (i.e. from αβrecursion).

import Printf:@sprintf
function Base.show(io::IO, ::MIME"text/plain", a::Array{Dict{State, T},1}) where T <: AbstractFloat
    for n in 1:length(a)
        write(io, "[n = $n]  \t")
        max = foldl(((sa,wa), (s,w)) -> wa < w ? (s,w) : (sa,wa), a[n]; init=first(a[n]))
        write(io, first(max) |> name)
        for (s, w) in sort(a[n]; by=x->name(x))
            write(io, "\t$(id(s)):$(name(s)) = $(@sprintf("%.3f", w))  ")
        end
        write(io, "\n")
    end
end

#######################################################################
# Major algorithms

#include("algorithms.jl")

#######################################################################
# Other

#include("../src/misc.jl")

end
