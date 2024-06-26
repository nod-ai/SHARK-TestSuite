# SHARK Tank tests

These test cases are exported from https://github.com/nod-ai/sharktank.

## Steps to add test cases

* Follow instructions in https://github.com/nod-ai/sharktank/blob/main/docs/model_cookbook.md
* Convert the exported `.mlir` to `.mlirbc`:

    ```bash
    iree-ir-tool cp file.mlir --emit-bytecode -o file.mlirbc
    ```

* Create a test_cases.json file with parameters, inputs, and outputs
  * Parameters can come from Hugging Face by using URL from "download file"
  * TODO: inputs and outputs should be exportable from sharktank/shortfin
    (or a script here - need to run the tokenizer and optionally populate the
    KV cache for some models)
