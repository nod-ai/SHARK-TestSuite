module {
  func.func @test_mvn_expanded(%arg0: !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.vtensor.literal(dense<2.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %1 = torch.vtensor.literal(dense<9.99999971E-10> : tensor<f32>) : !torch.vtensor<[],f32>
    %2 = torch.operator "onnx.ReduceMean"(%arg0) {torch.onnx.axes = [0 : si64, 2 : si64, 3 : si64]} : (!torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[1,3,1,1],f32>
    %3 = torch.operator "onnx.Pow"(%2, %0) : (!torch.vtensor<[1,3,1,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,1,1],f32>
    %4 = torch.operator "onnx.Pow"(%arg0, %0) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,3,3,1],f32>
    %5 = torch.operator "onnx.ReduceMean"(%4) {torch.onnx.axes = [0 : si64, 2 : si64, 3 : si64]} : (!torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[1,3,1,1],f32>
    %6 = torch.operator "onnx.Sub"(%5, %3) : (!torch.vtensor<[1,3,1,1],f32>, !torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[1,3,1,1],f32>
    %7 = torch.operator "onnx.Sqrt"(%6) : (!torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[1,3,1,1],f32>
    %8 = torch.operator "onnx.Sub"(%arg0, %2) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[3,3,3,1],f32>
    %9 = torch.operator "onnx.Add"(%7, %1) : (!torch.vtensor<[1,3,1,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[1,3,1,1],f32>
    %10 = torch.operator "onnx.Div"(%8, %9) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[1,3,1,1],f32>) -> !torch.vtensor<[3,3,3,1],f32>
    return %10 : !torch.vtensor<[3,3,3,1],f32>
  }
}

