
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate("./")

using JSON
@everywhere using MarkovModels
using ProgressMeter
@everywhere using Semirings
@everywhere using Serialization
@everywhere using SparseArrays
using TOML


@everywhere function statemap(fsm, numpdf)
    sparse(
        1:nstates(fsm)+1,
        map(x -> x[end], [val.(fsm.λ)..., (numpdf+1,)]),
        one(eltype(fsm.α)),
        nstates(fsm) + 1,
        numpdf + 1
    )
end

@everywhere function LinearFSM(K::Type{<:LogSemiring}, seq; init_silprob = 0, silprob = 0,
				   final_silprob = 0)
	arcs = []

	if init_silprob > 0
		init = [1 => K(log(init_silprob)), 2 => K(log(1 - init_silprob))]
		push!(arcs, (1, 2) => one(K))
		labels = [Label("<sil>"), Label(seq[1])]
		scount = 2
	else
		init = [1 => one(K)]
		labels = [Label(seq[1])]
		scount = 1
	end

	for (i, s) in enumerate(seq[2:end])
		if silprob > 0
			push!(arcs, (scount, scount + 1) => K(log(silprob)))
			push!(arcs, (scount, scount + 2) => K(log(1 - silprob)))
			push!(arcs, (scount + 1, scount + 2) => one(K))
			push!(labels, Label("<sil>"))
			push!(labels, Label(s))
			scount += 2
		else
			push!(arcs, (scount, scount + 1) => one(K))
			push!(labels, Label(s))
			scount += 1
		end
	end

	if final_silprob > 0
		final = [scount => K(log(1 - final_silprob)),
				 scount + 1 => one(K)]
		push!(arcs, (scount, scount + 1) => K(log(final_silprob)))
		push!(labels, Label("<sil>"))
	else
		final = [scount => one(K)]
	end

	FSM(init, arcs, final, labels)
end

function make_hmms(units, topojson)
	numpdf = 0
	jsondata = JSON.parsefile(topojson)
	nstates = length(jsondata["labels"])
	unitdict = Dict()
	open(units, "r") do f
        @showprogress for line in readlines(f)
			jsondata["labels"] = collect(numpdf+1:numpdf+nstates)
			unitdict[Label(String(strip(line)))] = FSM(json(jsondata))
			numpdf += nstates
		end
	end
	unitdict, numpdf
end

function make_lexicon(K, lexicon)
	lfsm = Dict()

	open(lexicon, "r") do f
        @showprogress for line in readlines(f)
			tokens = String.(split(line))
			word, pronun = tokens[1], tokens[2:end]
			word = Label(word)

			fsm = LinearFSM(K, pronun)
			if word in keys(lfsm)
				lfsm[word] = union(lfsm[word], fsm) |> minimize |> renorm
			else
				lfsm[word] = fsm
			end
		end
	end
	lfsm
end

function make_numerator_graphs(K, folder, text, lexicon, hmms, numpdf;
                               init_silprob, silprob, final_silprob,
                               ngram_order)

    @everywhere workers() mkpath(joinpath($folder, "$(myid() - 1)"))
    @everywhere workers() rm(joinpath($folder, "$(myid() - 1)", "fsm.scp"), force=true)
    @everywhere workers() rm(joinpath($folder, "$(myid() - 1)", "smap.scp"), force=true)
    ngrams = @showprogress @distributed (a, b) -> mergewith((x, y) -> x .+ y, a, b) for line = readlines(text)
		tokens = String.(split(line))
		uttid = tokens[1]
		seq = tokens[2:end]

		if isempty(seq)
		    Dict()
	    else
            seq = [Label(s) in keys(lexicon) ? s : "<unk>" for s in seq]

            G = LinearFSM(K, seq; init_silprob, silprob, final_silprob)
            GL = G ∘ lexicon
            GLH = GL ∘ hmms
            fsm_path = joinpath(folder, "$(myid() - 1)", uttid * ".fsm")
            serialize(fsm_path, GLH)
            smap_path = joinpath(folder, "$(myid() - 1)", uttid * ".smap")
            serialize(smap_path, statemap(GLH, numpdf))

            open(joinpath(folder, "$(myid() - 1)", "fsm.scp"), "a") do f
                println(f, uttid, " ", fsm_path)
            end
            open(joinpath(folder, "$(myid() - 1)", "smap.scp"), "a") do f
                println(f, uttid, " ", smap_path)
            end

            totalngramsum(GL, order = ngram_order)
        end
	end

	ngrams
end

# Look up the config file in the CONFIG environment variable.
config = TOML.parsefile(ENV["CONFIG"])

mkpath(config["supervision"]["folder"])

@info "Make the HMMs..."
hmms, numpdf = make_hmms(config["data"]["units"], config["supervision"]["topo"])

open(joinpath(config["supervision"]["folder"], "numpdf"), "w") do f
    println(f, "$numpdf")
end

# Extract the semiring type from the HMMs
K = eltype(collect(values(hmms))[1].α)

@info "Build the lexicon..."
lexicon = make_lexicon(K, config["data"]["lexicon"])


#@info "Build the numerator graphs (train) ($(Threads.nthreads()) threads)..."
@info "Build the numerator graphs (train) ($(nprocs() - 1) workers)..."
outfolder = joinpath(config["supervision"]["folder"], "numfsms", "train")
mkpath(outfolder)
ngrams = make_numerator_graphs(
    K,
    outfolder,
    config["data"]["traintext"],
    lexicon,
    hmms,
    numpdf;
    init_silprob = config["supervision"]["initial_silprob"],
    silprob = config["supervision"]["silprob"],
    final_silprob = config["supervision"]["final_silprob"],
    ngram_order = config["supervision"]["ngram_order"]
)

open(joinpath(outfolder, "fsm.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "fsm.scp")))
    end
end

open(joinpath(outfolder, "smap.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "smap.scp")))
    end
end


@info "Build the numerator graphs (dev) ($(nprocs() - 1) workers)..."
outfolder = joinpath(config["supervision"]["folder"], "numfsms", "dev")
mkpath(outfolder)
ngrams = make_numerator_graphs(
    K,
    outfolder,
    config["data"]["devtext"],
    lexicon,
    hmms,
    numpdf;
    init_silprob = config["supervision"]["initial_silprob"],
    silprob = config["supervision"]["silprob"],
    final_silprob = config["supervision"]["final_silprob"],
    ngram_order = config["supervision"]["ngram_order"]
)

open(joinpath(outfolder, "fsm.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "fsm.scp")))
    end
end

open(joinpath(outfolder, "smap.scp"), "w") do f
    for wid in workers()
        write(f, read(joinpath(outfolder, "$(wid - 1)", "smap.scp")))
    end
end

@info "Build the denominator graph..."
lmfsm = LanguageModelFSM(ngrams) ∘ hmms
serialize(joinpath(config["supervision"]["folder"], "denominator") * ".fsm",
          lmfsm)
serialize(joinpath(config["supervision"]["folder"], "denominator") * ".smap",
          statemap(lmfsm, numpdf))

