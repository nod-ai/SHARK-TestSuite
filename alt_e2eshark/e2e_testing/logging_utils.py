# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import traceback
import shutil
import subprocess
from e2e_testing.framework import result_comparison

from typing import List


def run_command_and_log(command: List[str], save_to: str, stage_name: str) -> None:
    """Runs command through subprocess.run and logs the command and error details (if present)"""
    # convert command list to a string
    script = subprocess.list2cmdline(command)
    # setup a commands subdirectory (if it doesn't exist)
    commands_dir = Path(save_to) / "commands"
    commands_dir.mkdir(exist_ok=True)
    # log the command
    commands_log = commands_dir / f"{stage_name}.commands.log"
    commands_log.write_text(script)
    # run the command
    ret = subprocess.run(script, shell=True, capture_output=True)
    # if an error occured, log stderr and raise exception
    if ret.returncode != 0:
        detail_dir = Path(save_to) / "detail"
        detail_dir.mkdir(exist_ok=True)
        detail_log = detail_dir / f"{stage_name}.detail.log"
        detail_log.write_text(ret.stderr.decode())
        error_msg = f"failure executing command:\n{script}"
        error_msg += f"Error detail in '{detail_log}'"
        raise RuntimeError(error_msg)


def log_result(result, log_dir, tol):
    # TODO: add more information for the result comparison (e.g., on verbose, add information on where the error is occuring, etc)
    summary = result_comparison(result, tol)
    num_match = 0
    num_total = 0
    for s in summary:
        num_match += s.sum().item()
        num_total += s.nelement()
    percent_correct = num_match / num_total if num_total != 0 else "N/A"
    with open(log_dir + "inference_comparison.log", "w+") as f:
        f.write(
            f"matching values with (rtol,atol) = {tol}: {num_match} of {num_total} = {percent_correct*100}%\n"
        )
        f.write(f"Test Result:\n{result}")
    return num_match == num_total


def log_exception(e: Exception, path: str, stage: str, name: str, verbose: bool):
    """generates a log for an exception generated during a testing stage"""
    log_filename = path + stage + ".log"
    base_str = f"Failed test at stage {stage} with exception:\n{e}\n"
    with open(log_filename, "w") as f:
        f.write(base_str)
        if verbose:
            print(f"\tFAILED ({stage})" + " " * 20)
            traceback.print_exception(e, file=f)
        else:
            print(f"FAILED: {name}")


def scan_dir_del_if_large(dir, size_MB):
    remove_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            curr_file = os.path.join(root, name)
            size_bytes = os.path.getsize(curr_file)
            if size_bytes >= size_MB * (10**6):
                remove_files.append(curr_file)
    for file in remove_files:
        os.remove(file)
    return remove_files


def scan_dir_del_mlir_vmfb(dir):
    removed_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            curr_file = os.path.join(root, name)
            if name.endswith(".mlir") or name.endswith(".vmfb"):
                removed_files.append(curr_file)
    for file in removed_files:
        os.remove(file)
    return removed_files


def scan_dir_del_not_logs(dir):
    removed_files = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            curr_file = os.path.join(root, name)
            if not name.endswith(".log") and not name.endswith(".json"):
                removed_files.append(curr_file)
    for file in removed_files:
        os.remove(file)
    return removed_files


def post_test_clean(log_dir, cleanup, verbose):
    match cleanup:
        case 0:
            return
        case 1:
            files = scan_dir_del_if_large(log_dir, 500)
        case 2:
            files = scan_dir_del_mlir_vmfb(log_dir)
        case 3:
            files = scan_dir_del_not_logs(log_dir)
        case 4:
            shutil.rmtree(Path(log_dir))
