module {
  func.func @test_center_crop_pad_crop_axes_hwc(%arg0: !torch.vtensor<[20,8,3],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[10,9,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.CenterCropPad"(%arg0, %arg1) {torch.onnx.axes = [0 : si64, 1 : si64]} : (!torch.vtensor<[20,8,3],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[10,9,3],f32>
    return %0 : !torch.vtensor<[10,9,3],f32>
  }
}

