module {
  func.func @test_maxpool_with_argmax_2d_precomputed_strides(%arg0: !torch.vtensor<[1,1,5,5],f32>) -> (!torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,1,2,2],si64>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0:2 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.storage_order = 1 : si64, torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,5,5],f32>) -> (!torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,1,2,2],si64>)
    return %0#0, %0#1 : !torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,1,2,2],si64>
  }
}

