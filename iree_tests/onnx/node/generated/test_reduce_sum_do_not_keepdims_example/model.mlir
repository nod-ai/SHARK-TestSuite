module {
  func.func @test_reduce_sum_do_not_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
    return %0 : !torch.vtensor<[3,2],f32>
  }
}

