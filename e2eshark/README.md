 # e2eshark framework-to-iree-to-inference tests

 This test suite enables developers to add small (operator level) to large (full model)
 end-2-end tests that compare output of running a model in a Framework 
 (e.g. Pytorch, ONNX) to the output of running the IREE commpiled artefact of 
 the same model on a target backend (e.g. CPU, AIE). If the difference in outputs
 is within a tolerable limit, then the test is reported as have passed, else the 
 test is reported as have failed. In case of a failing test, the stage of the 
 failure is reported. 

 The test suite organized starting with a framework name: pytorch, tensorflow, onnx. 
 For each framework category, multiple modes are tested. 

 - pytroch : starting model is a pytorch model
 - tensorflow : stating model is a tensorflow model (planned for later)
 - onnx : starting model is an onnx model generated using onnx python API or a onnx zip file
 
 Following, upto three, modes are supported based upon what is possible for a framework:

 - direct: Framework -> Framework graph (e.g. Torch Fx) -> Torch MLIR -> Compiled artefact -> Run target backend
 - onnx: Framework -> ONNX -> Import as Torch ONNX in Torch MLIR -> Torch MLIR -> Compiled artefact -> Run target backend
 - ort: Framework -> ONNX -> Load in IREE ONNX Runtime EP -> Compiled artefact -> Run target backend (planned for later)

 If Framework is 'onnx', then mode 'direct' will mean same as 'onnx'. For onnx/operators and onnx/combinations 
 the onnx model should be created using ONNX Python APIs. For onnx/models, a prebuilt onnx should be checked 
 in as a zip file.
 
 The target backend can be any IREE supported backend: llvm-cpu, amd-aie etc.

## Contents
 The contents are as below. Substitute 'Framework' by one of: pytorch, tensoflow, onnx
 - requirements.txt : 'pip install -r requirements.txt' to install needed additional 
                       packages not in your venv or conda. If you have a venv or conda 
                       environment for torch mlir or iree build, you can install 
                       this on top of that
 - run.py : Run 'python run.py --help' to learn about the script. This is the script to 
            run a specific test, all tests in a framework, all frameworks as per choice of a user
 - Framework/operators: This has operator level tests. Example: pytorch/operators/conv2d
 - Framework/combinations: This has small tests testing combination of operators such as 
                           pytorch/combinations/mlp testing a multi-layer perceptron which 
                           has torch.linear followed by torch.relu and repeated a few times
 - Framework/models: This has full model test. Since this is full model test, you may need 
                     necesary permsisions to download a model such as for llama2 you will 
                     need hugging face token. You should run 'huggingface-cli login' and 
                     enter the HF token before launching the run.py.
 - gold/passed.txt : This has list of tests that are passing as of today. If your check-in increases
                     passing tests, please update this. After your changes in torch MLIR or IREE, 
                     run all the tests upto inference and make sure that your are passing at least 
                     at the level of gold/passed.txt
 - tools/stubs/onnxmodel.py : This is concatenated to 'model.py' in test directory to form a 
                              runmodel.py for tests of framework 'onnx'
 - tools/stubs/pytorchmodel.py : This is concatenated to 'model.py' in test directory to form a 
                                 runmodel.py for tests of framework 'pytorch'
 - tools/onnxutil.py : Allows examining an ONNX (protobuf) file
 
 The logs are created as .log files in the test-run sub directory. Examine the logs to find and fix 
 cause of any failure. You can specify -r 'your dir name' to the run.py to name your test run directory 
 as per your choice. The default name for the run-directory is 'test-run'.

 You will be required to pass --hfhome argument to the run.py to point to a directory where 
 model weights etc. from Hugging Face will be downloaded. This can be large so set it to
 other than your home, preferably with 100 GB or more free space.

## Setting up

You will need to have a local build of torch MLIR and IREE. 

To get a local build of torch MLIR, follow:
https://github.com/llvm/torch-mlir/blob/main/docs/development.md

For torch MLIR build, build the torch_mlir python wheel and install it in your python env
(preferrred to ensure that you have the latest and greatest changes):

https://github.com/llvm/torch-mlir/blob/main/docs/development.md#build-python-packages 

Edit setup.py to insert following in cmake_args if you used clang to build torch-mlir 
```
 f"-DCMAKE_C_COMPILER=clang",
 f"-DCMAKE_CXX_COMPILER=clang++",
 ```
And then run following

```
CMAKE_GENERATOR=Ninja python setup.py bdist_wheel --dist-dir ./torch-mlir-wheel -v
pip uninstall torch-mlir
pip install torch-mlir-wheel/torch_mlir-0.0.1-cp310-cp310-linux_x86_64.whl
```
To get a local build of IREE, follow (prefer the clang building option over gnu):
https://iree.dev/building-from-source/getting-started 

If using AMD AIE target, build following in addition:
https://github.com/nod-ai/iree-amd-aie

To get this test repo:
```
git clone https://github.com/nod-ai/SHARK-TestSuite.git
cd e2eshark 
```
Set up python envonment, you can use venv or conda. 
Example to create a brand new conda env using python 3.10 is: 
```
conda create -n e2e python=3.10
conda activate e2e
pip install --upgrade pip

```
Make sure you do not skip the pip upgrade above as older pip may not able to handle pytorch deps
Then install needed packages as below:
```
pip install -r <your local torch MLIR repo>/requirements.txt
pip install -r <your local torch MLIR repo>/torchvision-requirements.txt
pip install <your local torch MLIR repo>/torch-mlir-wheel/torch_mlir-0.0.1-cp310-cp310-linux_x86_64.whl
pip install -r ./requirements.txt
```
Once setup, in any new shell you can activate the same env everytime you want to use it 
without needing to re-install requirements.txt's. 
Example:
```
conda activate e2e
```
## Examples

### Running tests

Example 1:
Run the tests in operators, combinations of the default framework (i.e. pytorch),
Use framework to onnx to torch MLIR path (--mode onnx) and run upto inference on default llvm-cpu backend
target, use four processor cores (--jobs 4) on your machine, generate report file after running
```
python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --report
```
You can see logs of test run inside test-run/'test sub-directory'. The test-run/statusreport.md and test-run/timereport.md
will show a nice table like below to give you detailed status of pass/fail of each stage. 
```
Test run status report

| test name                   | model-run | onnx-import | torch-mlir | iree-compile | inference |
| :-------------------------- | :-------- | :---------- | :--------- | :----------- | :-------- |
| pytorch/models/opt-1.3b     | passed    | passed      | passed     | passed       | failed    |
| pytorch/models/llama2-7b-hf | passed    | passed      | passed     | passed       | failed    |
| pytorch/models/resnet50     | passed    | passed      | passed     | passed       | passed    |
```

The test-run/passed.txt has list of all tests that passed and test-run/failed.txt has list of all 
the tests that failed. After you make changes in your source code to fix torch MLIR or IREE, you can 
re-run just the failing tests by simply passing test-run/failed.txt as an input like below:
```
python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --testsfile test-run/failed.txt
```
If you want to just generate report and skip run of tests then, you can pass --norun to skip running tests as below:
```
python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --testsfile test-run/failed.txt --norun --report
```

Example 2:
You can start from and run upto a stage. There are four stages: model-run, torch-mlir, iree-compile and inference
Say if you tested upto torch-mlir and do not want to test further and come back later and test torch-mlir onwards
later. As long as you have not destroyed the test-run dir, you can run following two at different times 
(first should have generted the torch MLIR for second one to resume successfully):
```
python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --frameworks pytorch onnx --runupto torch-mlir 

python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c 'path_to_your_torch_mlir_build_dir' -i 'path_to_your_iree_build_dir' --runfrom torch-mlir --runupto inference 

```
Example 3:
Run given test pytorch/models/opt-125M upto inference (default for --runupto) on target AMD AIE backend
```
python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c 'path_to_your_torch_mlir_build_dir' --frameworks onnx --runupto inference -i 'path_to_your_iree_build_dir' --tests pytorch/models/opt-125M --backend amd-aie
```

Example 4:
Print number of occureneces (frequency) of each operator kind in an ONNX file and print 
total count of operator instances

 ```
python ./tools/onnxutil.py onnx/models/resnet50_vaiq_int8/model.onnx -f
```
### Adding new tests

#### Adding test in framework pytorch
Let us say you wanted to add a new test to the framework "pytorch" to test maxpool  i.e. start with
pytorch model of maxpool, use run of the model as reference gold output and compare IREE compiled
output on your target backend. 

First google pytorch maxpool and read about the corresponding behavior of
maxpool operator in pytorch. For this example:
https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html provides description. 

Now take following steps:
1. Use an appropriate suffix to describe what you are intending to test say maxpool_2d_large
2. mkdir -p pytorch/operators/maxpool_2d_large
3. cd pytorch/operators/maxpool_2d_large
4. cp -pr pytorch/operators/conv2d/model.py .
5. modify model.py
6. You can run 'python ./model.py' to see input and output values printed to test 

Once your model.py is ready, you can go to root of the e2eshark test directory and run test as below 
   ```
   python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c "your torch mlir build dir" --tests pytorch/operators/maxpool_2d_large --mode direct --runupto torch-mlir --torchtolinalg
   ```
   Note that I did not specify -i for IREE build as I am running only upto torch-mlir. Also, I have added --torchtolinalg to make sure I test upto linalg lowerging as I am not running iree-compile

   Rerun above with --mode onnx if you want ONNX to get generated from pytorch and tested in addition.

   If you want to test upto inference, then provide your iree build in addition as -i option and run as

   ```
   python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c "your torch mlir build dir" -i 'path_to_your_iree_build_dir' --tests pytorch/operators/maxpool_2d_large --mode direct
   ```

   As before, run above with --mode onnx if you want ONNX to get generated from pytorch and tested in addition.

   cd into test-run/pytorch/operators/maxpool_2d_large and examine logs created. 
   Iterate over fixing your model.py and examing logs till you get it right. Given state of tools, tests may fail.
   If test fails because tool has bug, you are ready to add test. If test fails because of issue in model.py,
   you need to fix that before you can add test.
   
   Once you are satisfied, git add pytorch/operators/maxpool_2d_large/model.py, commit and publish it. 
   If that is a passing test and not in gold/passed.txt, add that passing test there as well

   #### Adding test in framework onnx

   Similarly to add a test in framework onnx for say cumsum operator, First google 
   pytorch ONNX.Cumsum and study behavior of the operator. For example, study
   https://onnx.ai/onnx/operators/onnx__CumSum.html . 

   If you do not already know how to write an ONNX model using ONNX Python API, study
   https://onnx.ai/onnx/intro/python.html


   Then take following steps:
   1. Use an appropriate suffix to describe what you are intending to test say cumsum_small
   2. mkdir -p onnx/operators/cumsum_small
   3. cd onnx/operators/cumsum_small
   4. cp -pr onnx/operators/gemm/model.py .
   5. modify model.py to model behavior of ONNX.cumsum
   6. run 'python ./model.py' to see input and output values, verify them to be as desired 
   
   Then test it as: 

   ```
   python ./run.py --hfhome 'YOUR_PATH'/HF_HOME -c "your torch mlir build dir" -i 'path_to_your_iree_build_dir' --tests onnx/operators/cumsum_small --mode direct --runupto inference --torchtolinalg
   ```

   Then follow steps similar to the one described above for pytorch framework to test more and add 
   the test to the repo.