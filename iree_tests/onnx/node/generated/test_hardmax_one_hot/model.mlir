module {
  func.func @test_hardmax_one_hot(%arg0: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Hardmax"(%arg0) : (!torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,4],f32> 
    return %0 : !torch.vtensor<[1,4],f32>
  }
}

