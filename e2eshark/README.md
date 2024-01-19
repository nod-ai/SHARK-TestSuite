 # End-to-end Pytorch tests for all shark target backends

 This test suite enables developers to add small to large (full model)
 end-2-end tests that compares result of running a model in PyTorch against
 running the commpiled model on a target backend. If the results are within
 a tolerable limit, then test passes else test fails. If there are compilation
 errors, then test will report the stage at which compilation failed.
 Three different paths are meant to be tested. All paths are not wired up
 yet -- work in progress.

 - Path 1: Pytorch -> Torch MLIR -> Compiled artefact -> Run target backend
 - Path 2: Pytorch -> Torch MLIR -> ONNX -> Import in Torch MLIR -> Compiled artefact -> Run target backend
 - Path 3: Pytorch -> Torch MLIR -> ONNX -> Import in IREE ONNX Runtime EP -> Compiled artefact -> Run target backend

 The target backend can be any supported hardware: cpu, amdaie, amdgpu etc.


 The contents are as explained below:
 - run.py : Run 'python ./run.py --help' to learn about the script. This is the script to run
          a specific test or all tests
 - test_template: This has a template for creating a new test. Copy this to a new test directory
          and modify the TEMPLATE_* identifiers to create a new test

 - operators: This has operator level test. This should test only
            one operator such as torch.Linear. Each test dir has a model.py which when run
            runs pytorch model, produces ONNX and/or torch MLIR as per option.
 - combinations: This has small test testing combination of operators such as toch.Linear
            followed by torch.Relu. Each test dir has a model.py which when run
            runs pytorch model, produces ONNX and/or torch MLIR as per option.
 - models: This has full model test. Each test dir has a model.py which when run
            runs pytorch model, produces ONNX and/or torch MLIR as per option. Since this is
            full model test, you may need necesary permsisions to download a model such as
            for llama2 you will need hugging face token. See the model.py for some guidance.

 This is a work in progress and will be fully wired up over time
