import os, time, glob, sys, zipfile
from multiprocessing import Pool
import argparse
import numpy as np
import shutil

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


def launchCommand(scriptcommand, commandslog):
    try:
        commandslog.write(scriptcommand)
        commandslog.write("\n")
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


def concatenateFiles(inpfile1, inpfile2, outfile):
    f1 = open(inpfile1, "r")
    f2 = open(inpfile2, "r")
    ofile = open(outfile, "w")
    ofile.write(f1.read() + f2.read())
    f1.close()
    f2.close()
    ofile.close()


def logAndReturn(commandslog, timelog, resultdict, retval):
    for i in resultdict:
        listitem = [i] + resultdict[i]
        print(listitem, file=timelog)
    timelog.close()
    commandslog.close()
    return retval


def unzipONNXFile(testName, abs_directory, unzipped_file_name):
    # Look for any unzipped file and if there is not already an unzipped file
    # then first time unzip it.
    abs_unzip_file_name = abs_directory + "/" + unzipped_file_name
    abs_zip_file_name = abs_unzip_file_name + ".zip"
    # this test dir does not have a zipped test file, so nothing to do
    if not os.path.exists(abs_zip_file_name):
        return 0
    # if not already unzipped, then
    if not os.path.exists(abs_unzip_file_name):
        if not os.access(abs_directory, os.W_OK):
            print(
                "The directory",
                abs_directory,
                "is not writeable. Could not unzip",
                abs_zip_file_name,
            )
            return 1
        with zipfile.ZipFile(abs_zip_file_name, "r") as zip_ref:
            zip_ref.extractall(abs_directory)

    return 0


def getTestKind(testName):
    # extract second last name in test name and if that is "models" and
    # it may have zipped onnx files, unzip them if not already done so
    second_last_name_inpath = os.path.split(os.path.split(testName)[0])[1]
    return second_last_name_inpath


def runInference(
    curphase,
    testName,
    args,
    vmfbfilename,
    modelinputfilename,
    goldoutputfilename,
    scriptcommand,
    commandslog,
    resultdict,
):
    # read the gold output produced by model
    logfilename = "inference.log"
    infoutputfilename = "inference.output.npy"

    modelinput = np.load(modelinputfilename)

    # numpy does not support bfloat16, so bfloat16 is cast to fp32 stored and
    # needs to be restored back
    # TODO: Does IREE handle bf16 as fp32 and back auomatically?

    goloutput = np.load(goldoutputfilename)

    if args.verbose:
        print("IREE run input:", modelinput)
        print("Framework gold:", goloutput)

    inputarg = ""
    # If there is no input the do not pass --input
    if modelinput.size > 0:
        inputarg = " --input=@" + modelinputfilename

    scriptcommand = (
        IREE_BUILD
        + "/tools/iree-run-module --module="
        + vmfbfilename
        + inputarg
        + " --output=@"
        + infoutputfilename
        + " > "
        + logfilename
        + " 2>&1"
    )

    start = time.time()
    if launchCommand(scriptcommand, commandslog):
        print("Test", testName, "failed[" + curphase + "]")
        return logAndReturn(commandslog, timelog, resultdict, 1)
    end = time.time()

    infoutput = np.load(infoutputfilename)
    if args.verbose:
        print("Framework gold output:", infoutput)
    # Adjust absolute tolerance and relative tolerance as needed
    # numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)[source]
    # if absolute(a - b) <= (atol + rtol * absolute(b)) is elementwise true
    # then numpy.allclose returns True
    rtol = 1e-05
    atol = 1e-08
    approximatematch = np.allclose(
        infoutput, goloutput, rtol=rtol, atol=atol, equal_nan=False
    )
    inferencematched = approximatematch
    if args.zerotolerance:
        # If each element matches exactly only then np.array_equal is true
        inferencematched = np.array_equal(infoutput, goloutput, equal_nan=False)
    if args.verbose:
        inerencelog = open(logfilename, "a")
        print("Gold reference:\n", goloutput, file=inerencelog)
        print("Output from target hardware:\n", infoutput, file=inerencelog)
        print("Difference: \n", goloutput - infoutput, file=inerencelog)
    if not inferencematched:
        failedinflog = open("failedinference.log", "w")
        print("Gold reference:\n", goloutput, file=failedinflog)
        print("Output from target hardware:\n", infoutput, file=failedinflog)
        print("Difference: \n", goloutput - infoutput, file=failedinflog)
        print("Test", testName, "failed[output-mismatch]")
        return logAndReturn(commandslog, timelog, resultdict, 1)

    resultdict[curphase] = ["passed", end - start]


def runTest(aTuple):
    curdir = os.getcwd()
    # Do not construct absolute path here as this will run
    # in a new process and cur dir may change over time giving
    # unpredicatble results
    (frameworkname, testName, args, script_dir, run_dir) = aTuple
    testargs = " --dtype " + args.dtype
    testRunDir = run_dir + "/" + testName
    modelname = os.path.basename(testName)
    modelinputfilename = testRunDir + "/" + modelname + "." + args.dtype + ".input.npy"
    goldoutputfilename = testRunDir + "/" + modelname + "." + args.dtype + ".output.npy"
    phases = ["model-run", "onnx-import", "torch-mlir", "ireecompile", "inference"]
    resultdict = {}
    for phase in phases:
        # Put status and time taken for each phase
        resultdict[phase] = ["notrun", 0.0]

    testAbsPath = script_dir + "/" + testName

    toolsDirAbsPath = script_dir + "/tools"
    stubrunmodelpy = toolsDirAbsPath + "stubs/pytorchmodel.py"
    modelpy = testAbsPath + "/model.py"
    # This is the generated runmodel.py which will be run
    runmodelpy = "runmodel.py"
    # For args.framework == onnx, onnx is starting point, so mode is
    # forced to onnx if it is direct
    mode = args.mode

    if args.verbose:
        print("Running:", testName, "[ Proc:", os.getpid(), "]")
    if changeToTestDir(testRunDir):
        return 1

    # Open files to log commands run and time taken
    commandslog = open("commands.log", "w")
    timelog = open("time.log", "w")

    # start phases[0]
    curphase = phases[0]
    stubrunmodelpy = toolsDirAbsPath + "/stubs/pytorchmodel.py"
    onnxfilename = modelname + "." + args.dtype + ".onnx"
    # Concatenate the testName model.py and tools/runmodel.py as run.py to
    # form runnable script.
    if frameworkname == "onnx":
        # For onnx, dierct and onnx means same as direct generates/has onnx itself
        if mode == "direct":
            mode = "onnx"
        stubrunmodelpy = toolsDirAbsPath + "/stubs/onnxmodel.py"
        onnxfilename = testAbsPath + "/model.onnx"
        if getTestKind(testName) == "models":
            # Create soft link to the model.onnx
            unzipONNXFile(testName, testAbsPath, "model.onnx")
            if not os.path.exists("model.onnx"):
                os.symlink(onnxfilename, "model.onnx")

    concatenateFiles(modelpy, stubrunmodelpy, runmodelpy)
    testargs += " --mode " + mode + " --outfileprefix " + modelname
    logfilename = modelname + ".log"
    scriptcommand = (
        "python " + runmodelpy + " " + testargs + " 1> " + logfilename + " 2>&1"
    )
    if args.verbose:
        print("Launching:", scriptcommand, "[ Proc:", os.getpid(), "]")
    start = time.time()
    if launchCommand(scriptcommand, commandslog):
        print("Test", testName, "failed[" + curphase + "]")
        return logAndReturn(commandslog, timelog, resultdict, 1)
    end = time.time()
    resultdict[curphase] = ["passed", end - start]

    torchmlirfilename = modelname + "." + args.dtype + ".pytorch.torch.mlir"

    if mode == "onnx" or mode == "ort":
        # start phases[1]
        curphase = phases[1]
        # Import ONNX into torch MLIR as torch.operator custom OP
        torchonnxfilename = modelname + "." + args.dtype + ".torch-onnx.mlir"
        logfilename = "torch-onnx.log"
        scriptcommand = (
            "python -m torch_mlir.tools.import_onnx "
            + onnxfilename
            + " -o "
            + torchonnxfilename
            + " 1> "
            + logfilename
            + " 2>&1"
        )
        start = time.time()
        if launchCommand(scriptcommand, commandslog):
            print("Test", testName, "failed[" + curphase + "]")
            return logAndReturn(commandslog, timelog, resultdict, 1)
        end = time.time()
        resultdict[curphase] = ["passed", end - start]

        # Lower torch ONNX to torch MLIR
        # start phases[2]
        curphase = phases[2]
        torchmlirfilename = modelname + "." + args.dtype + ".onnx.torch.mlir"
        logfilename = "onnxtotorch.log"
        scriptcommand = (
            TORCH_MLIR_BUILD
            + "/bin/torch-mlir-opt -convert-torch-onnx-to-torch "
            + torchonnxfilename
            + " > "
            + torchmlirfilename
            + " 2>"
            + logfilename
        )

        start = time.time()
        if launchCommand(scriptcommand, commandslog):
            print("Test", testName, "failed[" + curphase + "]")
            return logAndReturn(commandslog, timelog, resultdict, 1)
        end = time.time()
        resultdict[curphase] = ["passed", end - start]

    if args.upto == "torch-mlir":
        print("Test", testName, "passed")
        return logAndReturn(commandslog, timelog, resultdict, 0)

    # Compile torch MLIR using IREE to binary to target backend
    curphase = phases[3]
    vmfbfilename = modelname + "." + args.dtype + ".vfmb"
    logfilename = "ireecompile.log"
    scriptcommand = (
        IREE_BUILD
        + "/tools/iree-compile --iree-hal-target-backends="
        + args.backend
        + " "
        + torchmlirfilename
        + " > "
        + vmfbfilename
        + " 2>"
        + logfilename
    )
    start = time.time()
    if launchCommand(scriptcommand, commandslog):
        print("Test", testName, "failed[" + curphase + "]")
        return logAndReturn(commandslog, timelog, resultdict, 1)
    end = time.time()
    resultdict[curphase] = ["passed", end - start]

    if args.upto == "ireecompile":
        print("Test", testName, "passed")
        return logAndReturn(commandslog, timelog, resultdict, 0)

    # run inference now
    curphase = phases[4]
    if runInference(
        curphase,
        testName,
        args,
        vmfbfilename,
        modelinputfilename,
        goldoutputfilename,
        scriptcommand,
        commandslog,
        resultdict,
    ):
        return 1

    os.chdir(curdir)
    print("Test", testName, "passed")
    return logAndReturn(commandslog, timelog, resultdict, 0)


def runFrameworkTests(frameworkname, args, script_dir, run_dir):
    testsList = []
    poolSize = args.jobs
    if frameworkname == "tensorflow":
        print("The tensorflow is not supported yet.")
    if args.tests:
        testsList += args.tests

    if args.groups:
        if args.tests:
            print("Specific test(s) provided, test group will not be run")
        else:
            testsList += getTestsList(frameworkname, args.groups)
    # strip leading and trainling slashes
    for i, item in enumerate(testsList):
        testsList[i] = item.strip(os.sep)

    print("Running tests: ", testsList)
    uniqueTestList = []
    [uniqueTestList.append(test) for test in testsList if test not in uniqueTestList]
    if not uniqueTestList:
        print("No test specified.")
        sys.exit(1)
    tupleOfListArg = []
    # Create list of tuple(test, arg, run_dir) to allow launching tests in parallel
    [
        tupleOfListArg.append((frameworkname, test, args, script_dir, run_dir))
        for test in uniqueTestList
    ]
    if args.verbose:
        print("Following tests will be run:", uniqueTestList)

    with Pool(poolSize) as p:
        result = p.map_async(runTest, tupleOfListArg)
        result.wait()
        if args.verbose:
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
        choices=["direct", "onnx", "ort"],
        default="onnx",
        help="Use framework to torch MLIR, PyTorch to ONNX or ONNX plus ONNX RT stub flow",
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
        help="Run given specific test(s) only. Other test run options will be ignored.",
    )
    parser.add_argument(
        "-u",
        "--upto",
        choices=["torch-mlir", "ireecompile", "inference"],
        default="torch-mlir",
        help="Stop after genearting torch MLIR, or after IREE compilation, or go all the way to running inference.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print aditional messsages to show progress",
    )
    parser.add_argument(
        "-z",
        "--zerotolerance",
        action="store_true",
        default=False,
        help="Do not allow any tolerance in comparing results",
    )

    args = parser.parse_args()

    if args.torchmlirbuild:
        TORCH_MLIR_BUILD = args.torchmlirbuild
    TORCH_MLIR_BUILD = os.path.abspath(TORCH_MLIR_BUILD)
    if not os.path.exists(TORCH_MLIR_BUILD):
        print(
            "Torch MLIR build directory",
            TORCH_MLIR_BUILD,
            "does not exist.",
        )
        sys.exit(1)

    if args.ireebuild:
        IREE_BUILD = args.ireebuild
    if args.upto == "ireecompile" or args.upto == "inference":
        if not IREE_BUILD:
            print(
                "If --upto is 'ireecompile' or 'inference' then a valid IREE build is needed. Specify a valid IREE build directory using --ireebuild or set IREE_BUILD in run.py"
            )
            sys.exit(1)
        IREE_BUILD = os.path.abspath(IREE_BUILD)
        if not os.path.exists(IREE_BUILD):
            print("IREE build directory", IREE_BUILD, "does not exist.")
            sys.exit(1)

    run_dir = os.path.abspath(args.rundirectory)
    # Root dir where run.py is
    script_dir = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(run_dir):
        try:
            os.mkdir(run_dir)
        except OSError as errormsg:
            print("Could not make run directory", run_dir, " Error message: ", errormsg)
            sys.exit(1)
    print("Starting e2eshark tests. Using", args.jobs, "processes")
    if args.verbose:
        print("Test run with arguments: ", vars(args))
    print("Torch MLIR build:", TORCH_MLIR_BUILD)
    if IREE_BUILD:
        print("IREE build:", IREE_BUILD)
    print("Test run directory:", run_dir)
    for framework in args.frameworks:
        runFrameworkTests(framework, args, script_dir, run_dir)

    # When all processes are done, print
    print("Completed run of e2e shark tests")
