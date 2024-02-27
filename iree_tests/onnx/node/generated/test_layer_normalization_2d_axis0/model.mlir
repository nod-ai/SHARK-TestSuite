module {
  func.func @test_layer_normalization_2d_axis0(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[3,4],f32>, %arg2: !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0:3 = torch.operator "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>) -> (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>)
    return %0#0, %0#1, %0#2 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,1],f32>, !torch.vtensor<[1,1],f32>
  }
}

