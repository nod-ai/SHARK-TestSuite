# IREE Tests

This directory contains generated test suites for running through IREE's
compiler and runtime tools.

Each test suite has one folder per test program containing a few files:

```
[program name 1]/
  model.mlir
  input_0.npy
  output_0.npy
  test_data_flags.txt
```

Where:

* `model.mlir` is in a format that is ready for use with `iree-compile`
  (e.g. torch-mlir, stablehlo, tosa, linalg)
* `input_0.npy` and `output_0.npy` files correspond to any number of program
  inputs and outputs for one test case
* `test_data_flags.txt` is a flagfile for use with
  `iree-run-module --flagfile=test_data_flags.txt` of the format:

  ```
  --input=@input_0.npy
  --expected_output=@output_0.npy
  ```

## Running the test suite

Developer quickstart:

```bash
python -m pip install iree-compiler iree-runtime
python -m pip install -r ./iree_tests/onnx/requirements.txt

python ./iree_tests/onnx/import_tests.py
python ./iree_tests/compile_tests.py
python ./iree_tests/run_tests.py
```

Testing follows several stages:

1. Import
2. Compile
3. Run

### Importing tests

This could be either an online or an offline process. For now it is run
"offline" and the outputs are checked in to the repository for ease of use
in downstream projects and by developers who prefer to work directly with
`.mlir` files and native (C/C++) tools.

Each test suite or test case may also have its own import logic, with all test
suites converging onto the standard format described above by the end of this
stage.

> [!NOTE]
> Some test cases may fail to import.

### Compiling tests

The compile stage is performed for each desired target configuration.

> [!NOTE]
> Scripts are currently hardcoded to llvm-cpu -> local-task

A basic CPU configuration compiles like so:

```bash
iree-compile [model.mlir] --iree-hal-target-backends=llvm-cpu -o [model_cpu.vmfb]
```

and outputs a flagfile with contents like these:

```text
--module=[model_cpu.vmfb]
--device=local-task
```

> [!NOTE]
> Some test cases may fail to compile.

### Running tests

The run stage is performed on each desired device.

The CPU configuration above is run like so:

```bash
iree-run-module --flagfile=config_cpu_flags.txt --flagfile=test_data_flags.txt
```

## Available test suites

### ONNX test suite

The ONNX test suite is a conversion of the upstream test suite from
[onnx/`onnx/backend/test/`](../third_party/onnx/onnx/backend/test/):

* Python sources in [onnx/`onnx/backend/test/case/`](../third_party/onnx/onnx/backend/test/case)
* Generated `.onnx` and `[input,output]_[0-9]+.pb` files in [onnx/`onnx/backend/test/data/`](../third_party/onnx/onnx/backend/test/data)

The [`import_tests.py`](./onnx/import_tests.py) file walks test suites in the
'data' subdirectory and generates test cases in the format described above into
folders like [`./onnx/node/generated/`](./onnx/node/generated/).

The 'node' tests are for individual ONNX operators. The 'light', 'real',
'simple', and other test suites may also be interesting.
