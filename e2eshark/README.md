 # e2eshark framework-to-iree-to-inference tests

 This is work in progress.

 This test suite enables developers to add small to large (full model)
 end-2-end tests that compares result of running a model in Framework (e.g. Pytorch) 
 against running the commpiled model on a target backend using IREE. If the results 
 are within a tolerable limit, then test passes else test fails. If there are 
 compilation errors, then test will report the stage at which compilation failed.
 
 Test is organized starting with framework: pytorch, tensorflow, onnx
 For each framework category, multiple modes are tested. Following up-to three modes
 are supported based upon what is possible for a framework:

 - direct: Framework -> Torch MLIR -> Compiled artefact -> Run target backend
 - onnx: Framework -> ONNX -> Import as torch onnx in Torch MLIR -> Torch MLIR -> Compiled artefact -> Run target backend
 - ort: Framework -> ONNX -> Load in IREE ONNX Runtime EP -> Compiled artefact -> Run target backend

 If Framework is onnx, then mode 'direct' will mean same as 'onnx'
 The target backend can be any IREE supported backend: llvm-cpu, amd-aie etc.

## Contents
 The contents are as below. Substitute 'Framework' by one of: pytorch, tensoflow, onnx
 - requirements.txt : 'pip install -r requirements.txt' to install needed additional packages not in your venv or conda
 - run.py : Run 'python run.py --help' to learn about the script. This is the script to run a specific
            test, all tests in a framework, all frameworks as per choice of a user
 - Framework/test_template: This has a template for creating a new test. Copy this to a new test directory
          and modify the TEMPLATE_* identifiers to create a new test
 - Framework/operators: This has operator level test. example: pytorch/operators/conv2d
 - Framework/combinations: This has small test testing combination of operators such as 
            pytorch/combinations/mlp testing a multi layer perceptron which has torch.linear 
            followed by torch.relu and repeated a few times
 - Framework/models: This has full model test. Since this is full model test, you may need necesary 
            permsisions to download a model such as for llama2 you will need hugging face token. You
            should run 'huggingface-cli login' and enter the HF token before launching the test.
            Also set environment variable HF_HOME to a location with enough space. in bash, 
            export HF_HOME="your path"/HF_HOME

 - tools/onnxutil.py : Allows examining an ONNX protobuf file
 - tools/stubrunmodel.py : This is concatenated to 'model.py' in test directory to form a runmodel.py runnable model
 
 The logs are created as .log files in the test run sub directory. Examine the logs to find and fix 
 cause of any failure.

## Examples

Run the tests in models, operators, combinations of the default framework (i.e. pytorch),
use framework to onnx to torch MLIR path- and run up to hardware inference on default llvm-cpu
hardware target
```
python ./run.py --upto inference -j 4 -c ../../torch-mlir/build -g models combinations operators -i ../../iree-build --mode onnx
```

Print number of occureneces (frequency) of each operator kind in an ONNX file and print 
total count of operator instances

 ```
python ./tools/onnxutil.py onnx/models/resnet50_vaiq_int8/model.onnx -f
```

