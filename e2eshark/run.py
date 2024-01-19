import os, time, glob, sys, shutil
from multiprocessing import Pool
import argparse


def getTestsList(test_types):
    testsList = []
    for test_type in test_types:
        globpattern = test_type + "/*"
        testsList += glob.glob(globpattern)
    return testsList


def runTest(aList):
    print("Running test:", aList[0], "with args:", aList[1], "in process:", os.getpid())
    # Invoke the model.py first to generate the ONNX and/or Torch MLIR
    command = "python" + " " + aList[0] + "/model.py " + aList[1]
    os.system(command)

    # Import ONNX into torch MLIR
    # TODO: Add command

    # Compiler torch MLIR to compield artefact
    # TODO: Add command

    # Run test on target backend
    # TODO: Add command


if __name__ == "__main__":
    msg = "The run.py script to run e2e shark tests"
    parser = argparse.ArgumentParser(prog="run.py", description=msg, epilog="")

    parser.add_argument(
        "-d",
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Tensor datatype to use",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel processes to use for running tests",
    )
    parser.add_argument(
        "-r",
        "--runtests",
        nargs="*",
        default=["operators", "combinations"],
        choices=["operators", "combinations", "models"],
        help="Run given test categories",
    )
    parser.add_argument(
        "-t",
        "--testnames",
        nargs="*",
        help="Run specific tests.",
    )

    args = parser.parse_args()
    testArg = "--dtype " + args.dtype
    testsList = []
    poolSize = args.jobs
    if args.testnames:
        testsList += args.testnames
    if args.runtests:
        testsList += getTestsList(args.runtests)
    uniqueTestList = []
    [uniqueTestList.append(test) for test in testsList if test not in uniqueTestList]
    if not uniqueTestList:
        print("No test specified.")
        sys.exit(1)
    listOfListArg = []
    # Create list of pair(test, arg) to allow launching tests in parallel
    [listOfListArg.append([test, testArg]) for test in uniqueTestList]

    with Pool(poolSize) as p:
        result = p.map_async(runTest, listOfListArg)
        result.wait()
        print("All tasks submitted to process pool completed")

    # When all processes are done, print
    print("Completed run of e2e shark tests")
