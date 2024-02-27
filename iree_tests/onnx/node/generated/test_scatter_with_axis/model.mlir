module {
  func.func @test_scatter_with_axis(%arg0: !torch.vtensor<[1,5],f32>, %arg1: !torch.vtensor<[1,2],si64>, %arg2: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Scatter"(%arg0, %arg1, %arg2) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[1,5],f32>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,5],f32>
    return %0 : !torch.vtensor<[1,5],f32>
  }
}

