# SPDX-License-Identifier: MIT

"""
    fstprint(file, isyms=\"\", osyms=\"\")

Prints out binary FSTs in simple text format.

You can provide `isyms` and/or `osyms` mapping file (e.g. Kaldi's words.txt) to map input/output symbol IDs into symbols (e.g. words).
"""
fstprint(file; isyms="", osyms="") = `fstprint --isymbols=$isyms --osymbols=$osyms $file`


"""
    draw_fst(filename, isyms=\"\", osyms=\"\")

Wrapper around FST draw command, and graphviz dot, that displays the resulting visualisation.
"""
function draw_fst(filename; isyms="", osyms="")
    img = read(pipeline(
        `fstdraw --isymbols=$isyms --osymbols=$osyms --portrait $filename`,
        `dot -Tsvg`
    ), String)
    display("image/svg+xml", img)
end


function read_wfst(SF::Type{<:Semifield}, wfstfile; isyms="", osyms="")
    fst = read(fstprint(wfstfile; isyms=isyms, osyms=osyms), String);
    fst_finals = Dict()
    fst_arcs = Dict()

    for line in split(fst, '\n')
        arc = split(line, '\t')
        if length(arc) < 2
            continue
        elseif length(arc) < 4
            # terminal state
            final_node = parse(Int,arc[1])
            @assert final_node >= 0
            final_weight = parse(Float64, arc[2])
            fst_finals[final_node] = SF(-final_weight)
        else
            # non-terminal arc
            from_node = parse(Int,arc[1])
            to_node = parse(Int,arc[2])
            @assert from_node >= 0 && to_node >= 0
            isym = parse(Int, arc[3]) # transition ID
            osym = arc[4] # word
            weight = if length(arc) == 5 parse(Float64, arc[5]) else 0.0 end
            fst_arcs[from_node] = push!(get(fst_arcs, from_node, []), (to_node, isym, osym, SF(-weight)))
        end
    end

    return fst_arcs, fst_finals
end


function load_from_wfst!(fsm::AbstractFSM{SF}, wfstfile::String; symbol_table::String = "") where SF<:Semifield
    fst_arcs, fst_finals = read_wfst(SF, wfstfile, osyms=symbol_table)
    fst2fsm = Dict()

    s = addstate!(fsm, nothing, initweight = one(SF))
    fst2fsm[0] = s
    to_visit = [0]
    visited = []
    special_node = -1

    while !isempty(to_visit)
        @info "To visit: $(length(to_visit)), Visited: $(length(visited))"
        from_node = pop!(to_visit)
        push!(visited, from_node)
        for (to_node, isym, osym, weight) in get(fst_arcs, from_node, [])
            label = if string(isym) == "0" nothing else "$isym:$osym" end

            if !(to_node in visited) && !(to_node in to_visit)
                fst2fsm[to_node] = addstate!(fsm, label, finalweight=get(fst_finals, to_node, zero(SF)))
                push!(to_visit, to_node)
            elseif fst2fsm[to_node].label != label && from_node != to_node
                # different labels and not a self-loop
                # we need to create a different node, to distinguish it from the node
                @warn "$from_node -> $to_node: $(fst2fsm[to_node].label) != $label"
                @warn "... Crating a different node, to distinguish them"

                fst2fsm[special_node] = addstate!(fsm, label, finalweight=get(fst_finals, to_node, zero(SF)))
                addarc!(fsm, fst2fsm[from_node], fst2fsm[special_node], weight)
                fst_arcs[special_node] = map(get(fst_arcs, to_node, [])) do x
                    x[1] == to_node ? (special_node, x[2:end]...) : x
                end
                push!(to_visit, special_node)
                special_node -= 1
                continue
            end

            addarc!(fsm, fst2fsm[from_node], fst2fsm[to_node], weight)
        end
    end

    return fsm
end

"""
    load_from_kaldi(wfst_file, model_file, symbol_table= \"\")

Load Kaldi recognition network from `wfst_file`.

`model_file` (e.g. path/to/final.mdl) should contain Kaldi's transition model with the mapping transition ID -> pdf ID. This is used to create `MatrixFSM`.

`symbol_table` (e.g. path/to/graph/words.txt) is output symbol mapping table.
"""
function load_from_kaldi(wfst_file::String, kaldi_model::String; symbol_table::String = "")
    SF = LogSemifield{Float64}
    fsm = VectorFSM{SF}()
    MarkovModels.load_from_wfst!(fsm, wfst_file, symbol_table=symbol_table)

    trans_mdl = open("/mnt/matylda3/ikocour/tools/kaldi/egs/tidigits/s5/exp/mono0a/final.mdl") do fd
        Kaldi.is_binary(fd)
        Kaldi.load_transition_model(fd)
    end

    isselfloop(entry, hmm_state, transition_index) = begin
        transition_index <= length(entry[hmm_state].transitions) &&
            entry[hmm_state].transitions[transition_index].index == hmm_state
    end

    tid2pdf = Dict() # transition ID -> pdf ID mapping
    cur_transition_id = 0
    # Load tid2pdf
    for tstate in trans_mdl.tuples
        hmm_state = tstate.hmm_state+1
        entry = trans_mdl.topo[tstate.phone].entry
        transitions = entry[hmm_state].transitions
        prev_transition_id = cur_transition_id
        cur_transition_id += length(transitions)

        for tid in prev_transition_id+1:(cur_transition_id)
            if isselfloop(entry, hmm_state, tid - prev_transition_id) # self-loop
                tid2pdf[tid] = tstate.self_pdf + 1 # Kaldi index from 0
            else # forward-step
                tid2pdf[tid] = tstate.forward_pdf + 1 # Kaldi index from 0
            end
        end
    end

    fsm_no_eps = MarkovModels.remove_label(fsm, nothing)
    keyfn(label) = split(label, ":") |> first |> x -> parse(Int, x)
    return MatrixFSM(fsm_no_eps, tid2pdf, keyfn)
end

