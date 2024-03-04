module {
  func.func @test_floor(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Floor"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
}

