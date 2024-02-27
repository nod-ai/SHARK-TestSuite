module {
  func.func @test_not_2d(%arg0: !torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 1 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Not"(%arg0) : (!torch.vtensor<[3,4],i1>) -> !torch.vtensor<[3,4],i1>
    return %0 : !torch.vtensor<[3,4],i1>
  }
}

