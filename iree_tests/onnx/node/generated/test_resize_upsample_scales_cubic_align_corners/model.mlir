module {
  func.func public @test_resize_upsample_scales_cubic_align_corners(%arg0: !torch.vtensor<[1,1,4,4],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,8,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Resize"(%arg0, %none, %arg1) {torch.onnx.coordinate_transformation_mode = "align_corners", torch.onnx.mode = "cubic"} : (!torch.vtensor<[1,1,4,4],f32>, !torch.none, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,8,8],f32> 
    return %0 : !torch.vtensor<[1,1,8,8],f32>
  }
}

