module {
  func.func public @test_reduce_mean_default_axes_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ReduceMean"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[0],si64>) -> !torch.vtensor<[1,1,1],f32> 
    return %0 : !torch.vtensor<[1,1,1],f32>
  }
}

