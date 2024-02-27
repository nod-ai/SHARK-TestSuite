module {
  func.func @test_flatten_axis2(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[6,20],f32>
    return %0 : !torch.vtensor<[6,20],f32>
  }
}

