module {
  func.func @test_and3d(%arg0: !torch.vtensor<[3,4,5],i1>, %arg1: !torch.vtensor<[3,4,5],i1>) -> !torch.vtensor<[3,4,5],i1> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_version = 7 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.And"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],i1>) -> !torch.vtensor<[3,4,5],i1>
    return %0 : !torch.vtensor<[3,4,5],i1>
  }
}

