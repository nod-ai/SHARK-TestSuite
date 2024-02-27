module {
  func.func @test_gridsample_bilinear_align_corners_0_additional_1(%arg0: !torch.vtensor<[1,1,3,2],f32>, %arg1: !torch.vtensor<[1,2,4,2],f32>) -> !torch.vtensor<[1,1,2,4],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.GridSample"(%arg0, %arg1) {torch.onnx.align_corners = 0 : si64, torch.onnx.mode = "linear"} : (!torch.vtensor<[1,1,3,2],f32>, !torch.vtensor<[1,2,4,2],f32>) -> !torch.vtensor<[1,1,2,4],f32>
    return %0 : !torch.vtensor<[1,1,2,4],f32>
  }
}

