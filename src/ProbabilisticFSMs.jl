
#module MarkovModels
module ProbabilisticFSMs

#######################################################################
# FSM definition

export StateID
export initstateid
export finalstateid

export Label
export Pdfindex

export Link

export State
export isemitting

export FSM
export LinearFSM
export addstate!
export link!
export removestate!
export unlink!

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

#######################################################################
# Major algorithms

export PruningStrategy
export ThresholdPruning
export nopruning

# Baum-Welch algorithm
export αrecursion
export αβrecursion
export βrecursion
export viterbi

# FSM operations
export addselfloop!
export compose!
export determinize!
export minimize!
export removenilstates!
export weightnormalize!

include("algorithms.jl")

#######################################################################
# Pretty display functions

# Pretty display of the FSM in IJulia as a graph.
function Base.show(io, ::MIME"image/svg+xml", fsm::FSM)
    dotpath, dotfile = mktemp()
    svgpath, svgfile = mktemp()

    write(dotfile, "Digraph {\n")
    write(dotfile, "rankdir=LR;")

    for state in states(fsm)
        shape = ! isemitting(state) && ! islabeled(state) ? "point" : "circle"
        label = islabeled(state) ? "$(state.label)" : "$(state.pdfindex)"
        color = ! isemitting(state) && islabeled(state) ? "none" : "lightblue"
        write(dotfile, "$(state.id) [ shape=\"$(shape)\" label=\"$label \" style=filled fillcolor=$color ];\n")
    end

    for link in links(fsm)
        weight = round(link.weight, digits = 5)
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

# Pretty display the sparse matrix (i.e. from αβrecursion).
import Printf:@sprintf
function Base.show(
    io::IO,
    ::MIME"text/plain",
    a::Array{Dict{State, T},1}
) where T <: AbstractFloat

    for n in 1:length(a)
        write(io, "[n = $n]  \t")
        max = foldl(((sa,wa), (s,w)) -> wa < w ? (s,w) : (sa,wa), a[n]; init=first(a[n]))
        write(io, "$(first(max).id)")
        for (s, w) in sort(a[n]; by = x -> x.id)
            write(io, "\t$(s.id) = $(@sprintf("%.3f", w))  ")
        end
        write(io, "\n")
    end
end

#######################################################################
# Other

#include("../src/misc.jl")

end
