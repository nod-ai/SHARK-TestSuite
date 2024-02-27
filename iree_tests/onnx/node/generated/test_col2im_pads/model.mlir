module {
  func.func @test_col2im_pads(%arg0: !torch.vtensor<[1,5,15],f32>, %arg1: !torch.vtensor<[2],si64>, %arg2: !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Col2Im"(%arg0, %arg1, %arg2) {torch.onnx.pads = [0 : si64, 1 : si64, 0 : si64, 1 : si64]} : (!torch.vtensor<[1,5,15],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1,5,5],f32>
    return %0 : !torch.vtensor<[1,1,5,5],f32>
  }
}

