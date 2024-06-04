module {
  func.func public @test_flatten_axis0(%arg0: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[1,120],f32> 
    return %0 : !torch.vtensor<[1,120],f32>
  }
}

