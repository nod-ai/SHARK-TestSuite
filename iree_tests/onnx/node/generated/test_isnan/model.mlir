module {
  func.func @test_isnan(%arg0: !torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.IsNaN"(%arg0) : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[6],i1>
    return %0 : !torch.vtensor<[6],i1>
  }
}

