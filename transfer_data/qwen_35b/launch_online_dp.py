#!/usr/bin/env python3
"""
vLLM Online DP Launch Script (35B, single-machine 8-GPU, 4P + 4D)
"""
import argparse
import multiprocessing
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size.")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--dp-size-local", type=int, default=-1, help="Local data parallel size.")
    parser.add_argument("--dp-rank-start", type=int, default=0, help="Starting rank for data parallel.")
    parser.add_argument("--dp-address", type=str, required=True, help="IP address for data parallel master node.")
    parser.add_argument("--dp-rpc-port", type=str, default="12321", help="Port for data parallel master node.")
    parser.add_argument("--vllm-start-port", type=int, default=8000, help="Starting port for the engine.")
    parser.add_argument("--template", type=str, default="run_dp_template.sh", help="Template script name.")
    return parser.parse_args()

args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port
template = args.template

def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        f"./{template}",
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    subprocess.run(command, check=True)

if __name__ == "__main__":
    template_path = f"./{template}"
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    num_cards = dp_size_local * tp_size

    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        # Use dp_rank to assign GPUs globally across P and D
        visible_devices = ",".join(str(x) for x in range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
        process = multiprocessing.Process(
            target=run_command,
            args=(visible_devices, dp_rank, vllm_engine_port)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()