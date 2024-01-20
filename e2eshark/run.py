import os, time, glob, sys, shutil
from multiprocessing import Pool
import argparse

# Put full path to your Torch MLIR ( https://github.com/llvm/torch-mlir ) build
# This can be overwritten by command line
TORCH_MLIR_BUILD = ""
# Put full path to your IREE build dir
# if you are not targeting AMD AIE, then IREE build of https://github.com/openxla/iree is fine
# Else set it to build dir of your build of https://github.com/nod-ai/iree-amd-aie
# This can be overwritten by comamand line
IREE_BUILD = ""


def getTestsList(test_types):
    testsList = []
    for test_type in test_types:
        globpattern = test_type + "/*"
        testsList += glob.glob(globpattern)
    return testsList


def runTest(aList):
    testName = aList[0]
    arguments = aList[1]
    run_dir = os.path.abspath(aList[2] + "/" + testName)
    modelname = os.path.basename(testName)
    curdir = os.getcwd()
    print(
        "Running test:", modelname, "with args:", arguments, "in process:", os.getpid()
    )

    # Setup process invocation timeout
    scriptpath = "./" + testName + "/model.py "
    scriptcommand = os.path.abspath(scriptpath) + arguments
    print("Command:", scriptcommand)
    scriptcommand = "python " + scriptcommand + " > " + modelname + ".log"
    try:
        os.makedirs(run_dir, exist_ok=True)
        os.chdir(run_dir)
    except OSError as errormsg:
        print(
            "Could not make or change to test run directory",
            run_dir,
            " Error message: ",
            errormsg,
        )
        return

    try:
        print("Running: ", scriptcommand)
        os.system(scriptcommand)
    except OSError as errormsg:
        print(
            "Invoking ",
            scriptcommand,
            " failed:",
            errormsg,
        )
        return
    os.chdir(curdir)

    # TODO
    # Import ONNX into torch MLIR
    # python -m torch_mlir.tools.import_onnx <model>.onnx -o <model>.torch-onnx.mlir

    # Compiler torch MLIR to commpiled artefact
    # TORCH_MLIR_BUILD/bin/torch-mlir-opt -convert-torch-onnx-to-torch <model>.torch-onnx.mlir > <model>.onnx.torch.mlir

    # #Compile torch MLIR using IREE to binary to target backend
    # IREE_BUILD/tools/iree-compile --iree-hal-target-backends=<backend> <model>.pt.torch.mlir > <model>.<backend>.pt.vmfb
    # IREE_BUILD/tools/iree-compile --iree-hal-target-backends=<backend> <model>.onnx.torch.mlir > <model>.<backend>.onnx.vmfb

    # Run the copiled module on target and check correctness of result for each
    # /proj/gdba/kumar/nod/iree-build/tools/iree-run-module --module=<model>.<backend>.pt.vmfb --input="8x3xf32=0"
    # /proj/gdba/kumar/nod/iree-build/tools/iree-run-module --module=<model>.<model>.<backend>.onnx.vmfb --input="8x3xf32=0"


if __name__ == "__main__":
    msg = "The run.py script to run e2e shark tests"
    parser = argparse.ArgumentParser(prog="run.py", description=msg, epilog="")
    parser.add_argument(
        "-b",
        "--backend",
        choices=["llvm-cpu", "amd-aie", "rocm"],
        default="llvm-cpu",
        help="Target backend hardware",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Tensor datatype to use",
    )
    parser.add_argument(
        "-i",
        "--ireebuild",
        required=True,
        help="Path to the IREE build",
    )
    parser.add_argument(
        "-m",
        "--torchmlirbuild",
        required=True,
        help="Path to the torch-mlir build",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel processes to use for running tests",
    )
    parser.add_argument(
        "-g",
        "--groups",
        nargs="*",
        default=["operators", "combinations"],
        choices=["operators", "combinations", "models"],
        help="Run given group of tests",
    )
    parser.add_argument(
        "-t",
        "--tests",
        nargs="*",
        help="Run specific tests.",
    )
    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="Path to the torch-mlir build",
    )

    args = parser.parse_args()
    if args.torchmlirbuild:
        TORCH_MLIR_BUILD = args.torchmlirbuild
    if not os.path.exists(TORCH_MLIR_BUILD):
        print("IREE build directory", TORCH_MLIR_BUILD, "does not exist")
        sys.exit(1)

    if args.ireebuild:
        IREE_BUILD = args.ireebuild
    run_dir = args.rundirectory
    if not os.path.exists(run_dir):
        try:
            os.mkdir(run_dir)
        except OSError as errormsg:
            print("Could not make run directory", run_dir, " Error message: ", errormsg)
            sys.exit(1)

    if not os.path.exists(IREE_BUILD):
        print("IREE build directory", IREE_BUILD, "does not exist")
        sys.exit(1)

    testArg = "--dtype " + args.dtype
    testsList = []
    poolSize = args.jobs
    if args.tests:
        testsList += args.tests
        print("Running tests: ", testsList)
    if args.groups:
        if args.tests:
            print("Since specific tests were provided, test group will not be run")
        else:
            testsList += getTestsList(args.groups)
    uniqueTestList = []
    [uniqueTestList.append(test) for test in testsList if test not in uniqueTestList]
    if not uniqueTestList:
        print("No test specified.")
        sys.exit(1)
    listOfListArg = []
    # Create list of pair(test, arg) to allow launching tests in parallel
    [listOfListArg.append([test, testArg, run_dir]) for test in uniqueTestList]

    with Pool(poolSize) as p:
        result = p.map_async(runTest, listOfListArg)
        result.wait()
        print("All tasks submitted to process pool completed")

    # When all processes are done, print
    print("Completed run of e2e shark tests")
