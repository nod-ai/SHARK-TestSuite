module {
  func.func @test_sce_mean_weight_ii_3d_log_prob(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>, %arg2: !torch.vtensor<[5],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[3,5,2],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0:2 = torch.operator "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %arg2) {torch.onnx.ignore_index = 1 : si64, torch.onnx.reduction = "mean"} : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>, !torch.vtensor<[5],f32>) -> (!torch.vtensor<[],f32>, !torch.vtensor<[3,5,2],f32>)
    return %0#0, %0#1 : !torch.vtensor<[],f32>, !torch.vtensor<[3,5,2],f32>
  }
}

