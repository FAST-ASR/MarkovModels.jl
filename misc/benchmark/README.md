Benchmark between the implementation of the forward-backward algorithm provided by this package (`semifield`)
and the C++/CUDA implementation provided in the [pychain](https://github.com/YiwenShaoStephen/pychain)
python module (`pychain_log, pychain_leaky`). The forward-backward algorithm is calculated with a 3-gram
phonotactic language model (without any smoothing techniques) evaluated on the transcription of the 
Wall Street Journal corpus. The graph has ~3k states and ~50k arcs. The following results were 
obtained with an Intel(R) Xeon(R) CPU E5-2630 v2 @ 2.60GHz CPU and a NVIDIA GeForce GTX 1080 GPU.

```
lang    precision   seq. length   num. states   batch size  implementation  device  time (s)
julia   single      700           3032          128         semifield       cpu     340.8361165                                                                                                              
julia   single      700           3032          128         semifield       gpu     2.003256319
python  single      700           3032          128         pychain_log     cpu     576.4394602775574
python  single      700           3032          128         pychain_leaky   cpu     41.669665575027466
python  single      700           3032          128         pychain_log     gpu     19.98448657989502
python  single      700           3032          128         pychain_leaky   gpu     19.628916025161743
```

NOTE: `pychain_log` and `pychain_leaky` refers to the classical logarithmic and "leaky-hmm" implementaiton respectively.
      The "leaky-hmm" (described [here](https://www.isca-speech.org/archive/pdfs/interspeech_2016/povey16_interspeech.pdf))
      is an approximation of the classical forward-backward algorithm that adds an extra connection between all possible 
      states and run the inference in the probability domain. 
  
To replicate these results: 
  1. instantiate the enviroment by typing (in the `pkg` REPL):
     ```
     pkg> activate .
     (benchmark) pkg> instantiate
     ```
  2. run the Julia-based implementation:
     ```
     # This may take a while as we need to run the computations two times
     # to discard the compilation time. 
     $ julia --project benchmark.jl
     ```
  3. install pychain following the package [instructions](https://github.com/YiwenShaoStephen/pychain#installation-and-requirements)
  4. run the C++/CUDA based implementation:
     ```
     python benchmark.py
     ```
