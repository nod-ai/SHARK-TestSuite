module {
  func.func @test_scatternd_multiply(%arg0: !torch.vtensor<[4,4,4],f32>, %arg1: !torch.vtensor<[2,1],si64>, %arg2: !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.ScatterND"(%arg0, %arg1, %arg2) {torch.onnx.reduction = "mul"} : (!torch.vtensor<[4,4,4],f32>, !torch.vtensor<[2,1],si64>, !torch.vtensor<[2,4,4],f32>) -> !torch.vtensor<[4,4,4],f32>
    return %0 : !torch.vtensor<[4,4,4],f32>
  }
}

