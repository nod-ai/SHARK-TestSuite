module {
  func.func @test_gemm_transposeB(%arg0: !torch.vtensor<[3,6],f32>, %arg1: !torch.vtensor<[4,6],f32>, %arg2: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.transB = 1 : si64} : (!torch.vtensor<[3,6],f32>, !torch.vtensor<[4,6],f32>, !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[3,4],f32>
    return %0 : !torch.vtensor<[3,4],f32>
  }
}

