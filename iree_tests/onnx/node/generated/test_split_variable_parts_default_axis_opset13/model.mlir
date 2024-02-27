module {
  func.func @test_split_variable_parts_default_axis_opset13(%arg0: !torch.vtensor<[6],f32>, %arg1: !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[4],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0:2 = torch.operator "onnx.Split"(%arg0, %arg1) : (!torch.vtensor<[6],f32>, !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[4],f32>)
    return %0#0, %0#1 : !torch.vtensor<[2],f32>, !torch.vtensor<[4],f32>
  }
}

