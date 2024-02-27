module {
  func.func @test_gemm_all_attributes(%arg0: !torch.vtensor<[4,3],f32>, %arg1: !torch.vtensor<[5,4],f32>, %arg2: !torch.vtensor<[1,5],f32>) -> !torch.vtensor<[3,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) {torch.onnx.alpha = 2.500000e-01 : f32, torch.onnx.beta = 3.500000e-01 : f32, torch.onnx.transA = 1 : si64, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[4,3],f32>, !torch.vtensor<[5,4],f32>, !torch.vtensor<[1,5],f32>) -> !torch.vtensor<[3,5],f32>
    return %0 : !torch.vtensor<[3,5],f32>
  }
}

