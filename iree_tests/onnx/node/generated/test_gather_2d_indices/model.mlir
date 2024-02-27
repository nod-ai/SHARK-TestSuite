module {
  func.func @test_gather_2d_indices(%arg0: !torch.vtensor<[3,3],f32>, %arg1: !torch.vtensor<[1,2],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Gather"(%arg0, %arg1) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,3],f32>, !torch.vtensor<[1,2],si64>) -> !torch.vtensor<[3,1,2],f32>
    return %0 : !torch.vtensor<[3,1,2],f32>
  }
}

