module {
  func.func @test_sce_mean_weight_ii(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[3],si64>, %arg2: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %arg2) {torch.onnx.ignore_index = 0 : si64, torch.onnx.reduction = "mean"} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[3],si64>, !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32>
    return %0 : !torch.vtensor<[],f32>
  }
}

