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


function read_wfst(wfstfile; isyms="", osyms="")
	fst = read(fstprint(tidigit_hclg; isyms=isyms, osyms=osyms), String);
	fst_finals = Dict()
	fst_arcs = Dict()

	for line in split(fst, '\n')
	    arc = split(line, '\t')
	    if length(arc) < 2
	        continue
	    elseif length(arc) < 4
	        # terminal state
	        final_node = parse(Int,arc[1])
	        final_weight = parse(Float64, arc[2])
	        fst_finals[final_node] = SF(-final_weight)
	    else
	        # non-terminal arc
	        from_node = parse(Int,arc[1])
	        to_node = parse(Int,arc[2])
	        isym = parse(Int, arc[3]) # transition ID
	        osym = arc[4] # word
	        weight = if length(arc) == 5 parse(Float64, arc[5]) else 0.0 end
	        fst_arcs[from_node] = push!(get(fst_arcs, from_node, []), (to_node, isym, osym, SF(-weight)))
	    end
	end

	return fst_arcs, fst_finals
end


function load_from_wfst!(fsm::AbstractFSM{Tv}, wfstfile::String; symbol_table::String = "") where Tv
	fst_arcs, fst_finals = read_wfst(wfstfile, osyms=symbol_table)
	fst2fsm = Dict()

	s = addstate!(fsm, nothing, initweight = one(SF))
	fst2fsm[0] = s
	to_visit = [0]
	visited = []

	while !isempty(to_visit)
	    from_node = pop!(to_visit)
	    push!(visited, from_node)
	    for (to_node, isym, osym, weight) in fst_arcs[from_node]
	        if !(to_node in visited) && !(to_node in to_visit)
	            push!(to_visit, to_node)
                label = if string(isym) != "0" "$isym:$osym" else nothing end
	            fst2fsm[to_node] = addstate!(fsm, label, finalweight=get(fst_finals, to_node, zero(SF)))
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
function load_from_kaldi(wfst_file, kaldi_model; symbol_table::String = "")
    SF = LogSemifield{Float64}
    fsm = VectorFSM{SF}()
    MarkovModels.load_from_wfst!(fsm, wfst_file, symbol_table)

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
	return MatrixFSM(fsm, tid2pdf, keyfn)
end

