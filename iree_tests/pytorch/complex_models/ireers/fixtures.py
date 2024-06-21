# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import subprocess

class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess, cwd: str):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)  # Decode error or other: best we can do.

        super().__init__(
            f"Error invoking iree-compile\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n\n"
            f"Invoked with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )

class IreeRunException(Exception):
    """Runtime exception that preserves the command line and error output."""

    def __init__(
        self, process: subprocess.CompletedProcess, cwd: str, compile_cmd: str
    ):
        # iree-run-module sends output to both stdout and stderr
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)  # Decode error or other: best we can do.
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)  # Decode error or other: best we can do.

        super().__init__(
            f"Error invoking iree-run-module\n"
            f"Error code: {process.returncode}\n"
            f"Stderr diagnostics:\n{errs}\n"
            f"Stdout diagnostics:\n{outs}\n"
            f"Compiled with:\n"
            f"  cd {cwd} && {compile_cmd}\n\n"
            f"Run with:\n"
            f"  cd {cwd} && {process.args}\n\n"
        )

def get_compile_cmd(mlir_path: str, output_file: str, args: [str]):
    compile_args = [
        "iree-compile",
        mlir_path
    ]
    compile_args += args
    compile_args += ["-o", output_file]
    cmd = subprocess.list2cmdline(compile_args)
    return cmd

def iree_compile(mlir_path: str, output_file: str, args: [str], cwd):
    cmd = get_compile_cmd(mlir_path, output_file, args)
    logging.getLogger().info(
        f"Launching compile command:\n"  #
        f"cd {cwd} && {cmd}"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
    return_code = proc.returncode
    if return_code != 0:
        raise IreeCompileException(proc, cwd)

def iree_run_module(vmfb_name: str, args: [str], cwd, compile_cmd: str):
    run_args = [
        "iree-run-module",
        f"--module={vmfb_name}"
    ]
    run_args += args
    cmd = subprocess.list2cmdline(run_args)
    logging.getLogger().info(
        f"Launching run command:\n"  #
        f"cd {cwd} && {cmd}"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
    return_code = proc.returncode
    if return_code != 0:
        raise IreeRunException(proc, cwd, compile_cmd)
