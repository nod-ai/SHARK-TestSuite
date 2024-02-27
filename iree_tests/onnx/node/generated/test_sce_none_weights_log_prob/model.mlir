module {
  func.func @test_sce_none_weights_log_prob(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[3],si64>, %arg2: !torch.vtensor<[5],f32>) -> (!torch.vtensor<[3],f32>, !torch.vtensor<[3,5],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0:2 = torch.operator "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %arg2) {torch.onnx.reduction = "none"} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[3],si64>, !torch.vtensor<[5],f32>) -> (!torch.vtensor<[3],f32>, !torch.vtensor<[3,5],f32>)
    return %0#0, %0#1 : !torch.vtensor<[3],f32>, !torch.vtensor<[3,5],f32>
  }
}

