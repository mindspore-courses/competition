@ECHO OFF

REM Run the best Python performance profiler!


REM Step 0: run normally
python run_lprof.py


REM Step 1: put @profile decorator to the functions you wanna profile
REM e.g. `construct()` of following llama modules
REM you can search for "@profile" and uncomment these default places:
REM - benchmark
REM - llama_layer.LlamaFeedForward
REM - llama_transformer.LLamaAttention
REM - llama_transformer.LLamaDecodeLayer
REM - llama.LlamaModel
REM - llama.LlamaForCausalLM


REM Step 2: run the profiler
kernprof -l run_lprof.py


REM Step 3: read the report
python.exe -m line_profiler -rmt run_lprof.py.lprof
