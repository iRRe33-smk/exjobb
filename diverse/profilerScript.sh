 echo "Profiling script using profiler.py"
 python -m cProfile -s cumulative -o cprofile.stats profiler.py
 snakeviz cprofile.stats
 echo "Opening visualization using snakeviz"