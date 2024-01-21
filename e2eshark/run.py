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


def getTestsList(frameworkname, test_types):
    testsList = []
    for test_type in test_types:
        globpattern = frameworkname + "/" + test_type + "/*"
        testsList += glob.glob(globpattern)
    return testsList


def changeToTestDir(run_dir):
    try:
        # If directory does not exist, make it
        os.makedirs(run_dir, exist_ok=True)
        os.chdir(run_dir)
        return 0
    except OSError as errormsg:
        print(
            "Could not make or change to test run directory",
            run_dir,
            " Error message: ",
            errormsg,
        )
        return 1


def launchCommand(scriptcommand):
    print("Running command:", scriptcommand)
    try:
        os.system(scriptcommand)
        return 0
    except OSError as errormsg:
        print(
            "Invoking ",
            scriptcommand,
            " failed:",
            errormsg,
        )
        return 1


def concatenateFiles(inpfile1, inpfile2, outfile):
    f1 = open(inpfile1, "r")
    f2 = open(inpfile2, "r")
    ofile = open(outfile, "w")
    ofile.write(f1.read() + f2.read())


def runTest(aList):
    testName = aList[0]
    args = aList[1]
    testargs = " --dtype " + args.dtype
    run_dir = os.path.abspath(aList[2] + "/" + testName)
    modelname = os.path.basename(testName)
    # Root dir where run.py is
    scriptrootdirectory = os.path.dirname(os.path.realpath(__file__))
    testAbsPath = os.path.abspath(scriptrootdirectory + "/" + testName)
    toolsDirAbsPath = os.path.abspath(scriptrootdirectory + "/tools")
    stubrunmodelpy = toolsDirAbsPath + "/stubrunmodel.py"
    modelpy = testAbsPath + "/model.py"
    # This is the generated runmodel.py which will be run
    runmodelpy = "runmodel.py"

    curdir = os.getcwd()
    print("Running:", testName, "[ Proc:", os.getpid(), "]")
    if changeToTestDir(run_dir):
        return 1

    # Concatenate the testName model.py and tools/runmodel.py as run.py to
    # form runnable script.
    stubrunmodelpy = toolsDirAbsPath + "/stubrunmodel.py"
    concatenateFiles(modelpy, stubrunmodelpy, runmodelpy)

    # Run the model.py to find reference output, generate ONNX and torch MLIR
    # TODO decide based upon run to
    testargs += " --mode " + args.mode + " --outfileprefix " + modelname
    scriptcommand = "python " + runmodelpy + " " + testargs + " > " + modelname + ".log"
    if launchCommand(scriptcommand):
        return 1

    torchmlirfilename = modelname + "." + args.dtype + ".pytorch.torch.mlir"
    if args.mode == "onnx" or args.mode == "ort":
        # Import ONNX into torch MLIR as torch.operator custom OP
        onnxfilename = modelname + "." + args.dtype + ".onnx"
        torchonnxfilename = modelname + "." + args.dtype + ".torch-onnx.mlir"
        scriptcommand = (
            "python -m torch_mlir.tools.import_onnx "
            + onnxfilename
            + " -o "
            + torchonnxfilename
            + "> torch-onnx.log"
        )
        if launchCommand(scriptcommand):
            return 1

        # Lower torch ONNX to torch MLIR
        torchmlirfilename = modelname + "." + args.dtype + ".onnx.torch.mlir"
        scriptcommand = (
            TORCH_MLIR_BUILD
            + "/bin/torch-mlir-opt -convert-torch-onnx-to-torch "
            + torchonnxfilename
            + " > "
            + torchmlirfilename
        )
        if launchCommand(scriptcommand):
            return 1

    if args.upto == "torch-mlir":
        return 0

    # Compile torch MLIR using IREE to binary to target backend
    vmfbfilename = modelname + "." + args.dtype + ".vfmb"
    scriptcommand = (
        IREE_BUILD
        + "/tools/iree-compile --iree-hal-target-backends="
        + args.backend
        + " "
        + torchmlirfilename
        + " > "
        + vmfbfilename
    )
    if launchCommand(scriptcommand):
        return 1

    if args.upto == "ireecompile":
        return 0
    # TODO: Set the input string from the input dumped by the model
    # during earlier run
    inputstring = "8x3xf32=0"
    scriptcommand = (
        IREE_BUILD
        + "/tools/iree-run-module --module="
        + vmfbfilename
        + " --input="
        + inputstring
        + " > "
        + vmfbfilename
    )
    if launchCommand(scriptcommand):
        return 1

    os.chdir(curdir)
    return 0


def runFrameworkTests(frameworkname, args, run_dir):
    testsList = []
    poolSize = args.jobs
    if args.tests:
        testsList += frameworkname + "/" + args.tests
        print("Running tests: ", testsList)
    if args.groups:
        if args.tests:
            print("Since specific tests were provided, test group will not be run")
        else:
            testsList += getTestsList(frameworkname, args.groups)
    uniqueTestList = []
    [uniqueTestList.append(test) for test in testsList if test not in uniqueTestList]
    if not uniqueTestList:
        print("No test specified.")
        sys.exit(1)
    listOfListArg = []
    # Create list of tuple(test, arg, run_dir) to allow launching tests in parallel
    [listOfListArg.append([test, args, run_dir]) for test in uniqueTestList]

    with Pool(poolSize) as p:
        result = p.map_async(runTest, listOfListArg)
        result.wait()
        print("All tasks submitted to process pool completed")


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
        "-c",
        "--torchmlirbuild",
        required=True,
        help="Path to the torch-mlir build",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Tensor datatype to use",
    )
    parser.add_argument(
        "-f",
        "--frameworks",
        nargs="*",
        default=["pytorch"],
        choices=["pytorch", "onnx", "tensorflow"],
        help="Run tests for given framework(s)",
    )
    parser.add_argument(
        "-i",
        "--ireebuild",
        required=True,
        help="Path to the IREE build",
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
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel processes to use for running tests",
    )
    parser.add_argument(
        "-l",
        "--listfile",
        nargs="*",
        help="Run tests listed in given file only. Other test run options will be ignored.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["torch", "onnx", "ort"],
        default="onnx",
        help="Use PyTorch to torch MLIR, PyTorch to ONNX or ONNX plus ONNX RT stub flow",
    )
    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="Path to the torch-mlir build",
    )
    parser.add_argument(
        "-t",
        "--tests",
        nargs="*",
        help="Run given specific tests only. Other test run options will be ignored.",
    )
    parser.add_argument(
        "-u",
        "--upto",
        choices=["torch-mlir", "ireecompile", "inference"],
        default="torch-mlir",
        help="Stop after genearting torch MLIR, or after IREE compilation, or go all the way to running inference.",
    )

    args = parser.parse_args()
    if args.torchmlirbuild:
        TORCH_MLIR_BUILD = args.torchmlirbuild
    if not os.path.exists(TORCH_MLIR_BUILD):
        print("IREE build directory", TORCH_MLIR_BUILD, "does not exist")
        sys.exit(1)

    if args.ireebuild:
        IREE_BUILD = args.ireebuild
    TORCH_MLIR_BUILD = os.path.abspath(TORCH_MLIR_BUILD)
    IREE_BUILD = os.path.abspath(IREE_BUILD)

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

    for framework in args.frameworks:
        runFrameworkTests(framework, args, run_dir)

    # When all processes are done, print
    print("Completed run of e2e shark tests")
