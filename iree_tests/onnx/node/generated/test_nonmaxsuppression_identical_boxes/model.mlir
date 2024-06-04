module {
  func.func public @test_nonmaxsuppression_identical_boxes(%arg0: !torch.vtensor<[1,10,4],f32>, %arg1: !torch.vtensor<[1,1,10],f32>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1],f32>, %arg4: !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!torch.vtensor<[1,10,4],f32>, !torch.vtensor<[1,1,10],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1,3],si64> 
    return %0 : !torch.vtensor<[1,3],si64>
  }
}

