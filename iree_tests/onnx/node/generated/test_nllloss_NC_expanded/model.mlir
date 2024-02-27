module {
  func.func @test_nllloss_NC_expanded(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %3 = torch.operator "onnx.Unsqueeze"(%arg1, %2) : (!torch.vtensor<[3],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1],si64>
    %4 = torch.operator "onnx.GatherElements"(%arg0, %3) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[3,1],si64>) -> !torch.vtensor<[3,1],f32>
    %5 = torch.operator "onnx.Neg"(%4) : (!torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,1],f32>
    %6 = torch.operator "onnx.Slice"(%5, %0, %1, %1) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1],f32>
    %7 = torch.operator "onnx.Squeeze"(%6, %2) : (!torch.vtensor<[3,1],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],f32>
    return %7 : !torch.vtensor<[3],f32>
  }
}

