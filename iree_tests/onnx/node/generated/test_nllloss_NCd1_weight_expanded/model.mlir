module {
  func.func @test_nllloss_NCd1_weight_expanded(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>, %arg2: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %3 = torch.operator "onnx.Unsqueeze"(%arg1, %2) : (!torch.vtensor<[3,2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],si64>
    %4 = torch.operator "onnx.GatherElements"(%arg0, %3) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,1,2],si64>) -> !torch.vtensor<[3,1,2],f32>
    %5 = torch.operator "onnx.Neg"(%4) : (!torch.vtensor<[3,1,2],f32>) -> !torch.vtensor<[3,1,2],f32>
    %6 = torch.operator "onnx.Slice"(%5, %0, %1, %1) : (!torch.vtensor<[3,1,2],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32>
    %7 = torch.operator "onnx.Gather"(%arg2, %arg1) : (!torch.vtensor<[5],f32>, !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[3,2],f32>
    %8 = torch.operator "onnx.Squeeze"(%6, %2) : (!torch.vtensor<[3,1,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32>
    %9 = torch.operator "onnx.Mul"(%8, %7) : (!torch.vtensor<[3,2],f32>, !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[3,2],f32>
    %10 = torch.operator "onnx.ReduceSum"(%9) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2],f32>) -> !torch.vtensor<[],f32>
    %11 = torch.operator "onnx.ReduceSum"(%7) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2],f32>) -> !torch.vtensor<[],f32>
    %12 = torch.operator "onnx.Div"(%10, %11) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32>
    return %12 : !torch.vtensor<[],f32>
  }
}

