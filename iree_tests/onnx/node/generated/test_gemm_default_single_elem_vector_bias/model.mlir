module {
  func.func @test_gemm_default_single_elem_vector_bias(%arg0: !torch.vtensor<[3,7],f32>, %arg1: !torch.vtensor<[7,3],f32>, %arg2: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Gemm"(%arg0, %arg1, %arg2) : (!torch.vtensor<[3,7],f32>, !torch.vtensor<[7,3],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,3],f32>
    return %0 : !torch.vtensor<[3,3],f32>
  }
}

