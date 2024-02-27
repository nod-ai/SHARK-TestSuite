module {
  func.func @test_nllloss_NCd1d2_with_weight_reduction_sum_ii_expanded(%arg0: !torch.vtensor<[3,5,6,6],f32>, %arg1: !torch.vtensor<[3,6,6],si64>, %arg2: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %1 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %2 = torch.vtensor.literal(dense<1> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %3 = torch.operator "onnx.Unsqueeze"(%arg1, %2) : (!torch.vtensor<[3,6,6],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6],si64>
    %4 = torch.vtensor.literal(dense<0> : tensor<1xsi64>) : !torch.vtensor<[1],si64>
    %5 = torch.operator "onnx.Sub"(%3, %3) : (!torch.vtensor<[3,1,6,6],si64>, !torch.vtensor<[3,1,6,6],si64>) -> !torch.vtensor<[3,1,6,6],si64>
    %6 = torch.operator "onnx.Cast"(%3) {torch.onnx.to = 7 : si64} : (!torch.vtensor<[3,1,6,6],si64>) -> !torch.vtensor<[3,1,6,6],si64>
    %7 = torch.operator "onnx.Equal"(%6, %4) : (!torch.vtensor<[3,1,6,6],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6],i1>
    %8 = torch.operator "onnx.Where"(%7, %5, %3) : (!torch.vtensor<[3,1,6,6],i1>, !torch.vtensor<[3,1,6,6],si64>, !torch.vtensor<[3,1,6,6],si64>) -> !torch.vtensor<[3,1,6,6],si64>
    %9 = torch.operator "onnx.GatherElements"(%arg0, %8) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,5,6,6],f32>, !torch.vtensor<[3,1,6,6],si64>) -> !torch.vtensor<[3,1,6,6],f32>
    %10 = torch.vtensor.literal(dense<0.000000e+00> : tensor<1xf32>) : !torch.vtensor<[1],f32>
    %11 = torch.operator "onnx.Where"(%7, %10, %9) : (!torch.vtensor<[3,1,6,6],i1>, !torch.vtensor<[1],f32>, !torch.vtensor<[3,1,6,6],f32>) -> !torch.vtensor<[3,1,6,6],f32>
    %12 = torch.operator "onnx.Neg"(%11) : (!torch.vtensor<[3,1,6,6],f32>) -> !torch.vtensor<[3,1,6,6],f32>
    %13 = torch.operator "onnx.Slice"(%12, %0, %1, %1) : (!torch.vtensor<[3,1,6,6],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6],f32>
    %14 = torch.operator "onnx.Gather"(%arg2, %8) : (!torch.vtensor<[5],f32>, !torch.vtensor<[3,1,6,6],si64>) -> !torch.vtensor<[3,1,6,6],f32>
    %15 = torch.operator "onnx.Where"(%7, %10, %14) : (!torch.vtensor<[3,1,6,6],i1>, !torch.vtensor<[1],f32>, !torch.vtensor<[3,1,6,6],f32>) -> !torch.vtensor<[3,1,6,6],f32>
    %16 = torch.operator "onnx.Squeeze"(%15, %2) : (!torch.vtensor<[3,1,6,6],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,6,6],f32>
    %17 = torch.operator "onnx.Squeeze"(%13, %2) : (!torch.vtensor<[3,1,6,6],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,6,6],f32>
    %18 = torch.operator "onnx.Mul"(%17, %16) : (!torch.vtensor<[3,6,6],f32>, !torch.vtensor<[3,6,6],f32>) -> !torch.vtensor<[3,6,6],f32>
    %19 = torch.operator "onnx.ReduceSum"(%18) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,6,6],f32>) -> !torch.vtensor<[],f32>
    return %19 : !torch.vtensor<[],f32>
  }
}

