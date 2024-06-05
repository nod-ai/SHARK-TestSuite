import os
from collections import namedtuple

benchmark_dir = os.path.dirname(os.path.realpath(__file__))

def launchCommand(scriptcommand):
    print("Launching:", scriptcommand, "[ Proc:", os.getpid(), "]")
    try:
        ret = os.system(scriptcommand)
        return ret
    except OSError as errormsg:
        print(
            "Invoking ",
            scriptcommand,
            " failed:",
            errormsg,
        )
        return 1

def run_sdxl_rocm_benchmark():
    scriptcommand = (
        "bash "
        + f"{benchmark_dir}/benchmark_sdxl_rocm.sh"
    )
    return launchCommand(scriptcommand)

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)

def decode_output(bench_lines):
    benchmark_results = []
    for line in bench_lines[3:]:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )
    return benchmark_results

def test_sdxl_rocm_benchmark(goldentime):
    if run_sdxl_rocm_benchmark():
        print("Running SDXL ROCm benchmark failed. Exiting")
        return
    with open(f"{benchmark_dir}/benchmark_out_rocm.txt") as f:
        bench_lines = f.readlines()
    benchmark_results = decode_output(bench_lines)
    benchmark_mean_time = int(benchmark_results[15].time.split()[0])
    assert benchmark_mean_time < goldentime
