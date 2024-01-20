 # End-to-end Pytorch tests for all shark target backends

 This is work in progress and not fully ready.

 This test suite enables developers to add small to large (full model)
 end-2-end tests that compares result of running a model in Framework (Pytorch) 
 against running the commpiled model on a target backend using IREE. If the results 
 are within a tolerable limit, then test passes else test fails. If there are 
 compilation errors, then test will report the stage at which compilation failed.
 
 For each framework category, multiple flows are tested. The flow is named by how input
 to IREE is generated. For example, for PyTorch, the torch MLIR for IREE can be generated
 in following ways

 - direct: Pytorch -> Torch MLIR -> Compiled artefact -> Run target backend
 - torch-onnx: Pytorch -> ONNX -> Import as torch onnx in Torch MLIR -> Torch MLIR -> Compiled artefact -> Run target backend
 - ort-ep: Pytorch -> ONNX -> Load in IREE ONNX Runtime EP -> Compiled artefact -> Run target backend

 The target backend can be any supported hardware: cpu, amdaie, amdgpu etc.


 The contents are as explained below. <framework> can be pytorch, tensorflow, onnx
 - run.py : Run 'python run.py --help' to learn about the script. This is the script to run a specific
            test, all tests in a framework, all frameworks as per choice of a user
 - <framework>/test_template: This has a template for creating a new test. Copy this to a new test directory
          and modify the TEMPLATE_* identifiers to create a new test

 - <framework>/operators: This has operator level test. 
 - <framework>/combinations: This has small test testing combination of operators such as toch.Linear
            followed by torch.Relu. 
 - <framework>/models: This has full model test. Since this is full model test, you may need necesary 
            permsisions to download a model such as for llama2 you will need hugging face token. 
            See the test directory for guidance.

