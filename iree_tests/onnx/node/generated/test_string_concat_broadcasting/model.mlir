module {
  func.func public @test_string_concat_broadcasting(%arg0: !torch.vtensor<[3],!torch.str>, %arg1: !torch.vtensor<[1],!torch.str>) -> !torch.vtensor<[3],!torch.str> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.StringConcat"(%arg0, %arg1) : (!torch.vtensor<[3],!torch.str>, !torch.vtensor<[1],!torch.str>) -> !torch.vtensor<[3],!torch.str> 
    return %0 : !torch.vtensor<[3],!torch.str>
  }
}

