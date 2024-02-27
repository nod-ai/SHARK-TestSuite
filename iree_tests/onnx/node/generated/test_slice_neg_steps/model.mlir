module {
  func.func @test_slice_neg_steps(%arg0: !torch.vtensor<[20,10,5],f32>, %arg1: !torch.vtensor<[3],si64>, %arg2: !torch.vtensor<[3],si64>, %arg3: !torch.vtensor<[3],si64>, %arg4: !torch.vtensor<[3],si64>) -> !torch.vtensor<[19,3,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[20,10,5],f32>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[19,3,2],f32>
    return %0 : !torch.vtensor<[19,3,2],f32>
  }
}

