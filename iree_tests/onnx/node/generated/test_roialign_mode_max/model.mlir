module {
  func.func @test_roialign_mode_max(%arg0: !torch.vtensor<[1,1,10,10],f32>, %arg1: !torch.vtensor<[3,4],f32>, %arg2: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.RoiAlign"(%arg0, %arg1, %arg2) {torch.onnx.coordinate_transformation_mode = "output_half_pixel", torch.onnx.mode = "max", torch.onnx.output_height = 5 : si64, torch.onnx.output_width = 5 : si64, torch.onnx.sampling_ratio = 2 : si64, torch.onnx.spatial_scale = 1.000000e+00 : f32} : (!torch.vtensor<[1,1,10,10],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,1,5,5],f32>
    return %0 : !torch.vtensor<[3,1,5,5],f32>
  }
}

