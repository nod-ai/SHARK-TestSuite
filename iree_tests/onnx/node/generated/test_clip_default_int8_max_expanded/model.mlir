module {
  func.func @test_clip_default_int8_max_expanded(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Less"(%arg1, %arg0) : (!torch.vtensor<[],si8>, !torch.vtensor<[3,4,5],si8>) -> !torch.vtensor<[3,4,5],i1>
    %1 = torch.operator "onnx.Where"(%0, %arg1, %arg0) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[],si8>, !torch.vtensor<[3,4,5],si8>) -> !torch.vtensor<[3,4,5],si8>
    return %1 : !torch.vtensor<[3,4,5],si8>
  }
}

