module {
  func.func public @test_layer_normalization_3d_axis2_epsilon(%arg0: !torch.vtensor<[2,3,5],f32>, %arg1: !torch.vtensor<[5],f32>, %arg2: !torch.vtensor<[5],f32>) -> (!torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,3,1],f32>, !torch.vtensor<[2,3,1],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = call @"('LayerNormalization', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], input: \22X\22\0Ainput: \22W\22\0Ainput: \22B\22\0Aoutput: \22Y\22\0Aoutput: \22Mean\22\0Aoutput: \22InvStdDev\22\0Aop_type: \22LayerNormalization\22\0Aattribute {\0A  name: \22axis\22\0A  i: 2\0A  type: INT\0A}\0Aattribute {\0A  name: \22epsilon\22\0A  f: 0.1\0A  type: FLOAT\0A}\0A)"(%arg0, %arg1, %arg2) : (!torch.vtensor<[2,3,5],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32>) -> (!torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,3,1],f32>, !torch.vtensor<[2,3,1],f32>)
    return %0#0, %0#1, %0#2 : !torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,3,1],f32>, !torch.vtensor<[2,3,1],f32>
  }
  func.func private @"('LayerNormalization', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], input: \22X\22\0Ainput: \22W\22\0Ainput: \22B\22\0Aoutput: \22Y\22\0Aoutput: \22Mean\22\0Aoutput: \22InvStdDev\22\0Aop_type: \22LayerNormalization\22\0Aattribute {\0A  name: \22axis\22\0A  i: 2\0A  type: INT\0A}\0Aattribute {\0A  name: \22epsilon\22\0A  f: 0.1\0A  type: FLOAT\0A}\0A)"(%arg0: !torch.vtensor<[2,3,5],f32>, %arg1: !torch.vtensor<[5],f32>, %arg2: !torch.vtensor<[5],f32>) -> (!torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,3,1],f32>, !torch.vtensor<[2,3,1],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e-01> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Cast"(%0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[2,3,5],f32>) -> !torch.vtensor<[3],si64> 
    %3 = torch.operator "onnx.Size"(%2) : (!torch.vtensor<[3],si64>) -> !torch.vtensor<[],si64> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %6 = torch.operator "onnx.Slice"(%2, %4, %5) : (!torch.vtensor<[3],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2],si64> 
    %7 = torch.operator "onnx.Sub"(%3, %5) : (!torch.vtensor<[],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    %8 = torch.operator "onnx.ConstantOfShape"(%7) {torch.onnx.value = dense<1> : tensor<1xsi64>} : (!torch.vtensor<[1],si64>) -> !torch.vtensor<[1],si64> 
    %9 = torch.operator "onnx.Concat"(%6, %8) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],si64> 
    %10 = torch.operator "onnx.Flatten"(%arg0) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[2,3,5],f32>) -> !torch.vtensor<[6,5],f32> 
    %11 = torch.operator "onnx.Cast"(%10) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[6,5],f32>) -> !torch.vtensor<[6,5],f32> 
    %12 = torch.operator "onnx.ReduceMean"(%11) {torch.onnx.axes = [1 : si64]} : (!torch.vtensor<[6,5],f32>) -> !torch.vtensor<[6,1],f32> 
    %13 = torch.operator "onnx.Mul"(%11, %11) : (!torch.vtensor<[6,5],f32>, !torch.vtensor<[6,5],f32>) -> !torch.vtensor<[6,5],f32> 
    %14 = torch.operator "onnx.ReduceMean"(%13) {torch.onnx.axes = [1 : si64]} : (!torch.vtensor<[6,5],f32>) -> !torch.vtensor<[6,1],f32> 
    %15 = torch.operator "onnx.Mul"(%12, %12) : (!torch.vtensor<[6,1],f32>, !torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,1],f32> 
    %16 = torch.operator "onnx.Sub"(%14, %15) : (!torch.vtensor<[6,1],f32>, !torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,1],f32> 
    %17 = torch.operator "onnx.Add"(%16, %1) : (!torch.vtensor<[6,1],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[6,1],f32> 
    %18 = torch.operator "onnx.Sqrt"(%17) : (!torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,1],f32> 
    %19 = torch.operator "onnx.Sub"(%11, %12) : (!torch.vtensor<[6,5],f32>, !torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,5],f32> 
    %20 = torch.operator "onnx.Div"(%19, %18) : (!torch.vtensor<[6,5],f32>, !torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,5],f32> 
    %21 = torch.operator "onnx.Cast"(%20) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[6,5],f32>) -> !torch.vtensor<[6,5],f32> 
    %22 = torch.operator "onnx.Flatten"(%arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[5],f32>) -> !torch.vtensor<[1,5],f32> 
    %23 = torch.operator "onnx.Mul"(%21, %22) : (!torch.vtensor<[6,5],f32>, !torch.vtensor<[1,5],f32>) -> !torch.vtensor<[6,5],f32> 
    %24 = torch.operator "onnx.Flatten"(%arg2) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[5],f32>) -> !torch.vtensor<[1,5],f32> 
    %25 = torch.operator "onnx.Add"(%23, %24) : (!torch.vtensor<[6,5],f32>, !torch.vtensor<[1,5],f32>) -> !torch.vtensor<[6,5],f32> 
    %26 = torch.operator "onnx.Reshape"(%25, %2) : (!torch.vtensor<[6,5],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,5],f32> 
    %27 = torch.operator "onnx.Reciprocal"(%18) : (!torch.vtensor<[6,1],f32>) -> !torch.vtensor<[6,1],f32> 
    %28 = torch.operator "onnx.Reshape"(%12, %9) : (!torch.vtensor<[6,1],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,1],f32> 
    %29 = torch.operator "onnx.Reshape"(%27, %9) : (!torch.vtensor<[6,1],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[2,3,1],f32> 
    return %26, %28, %29 : !torch.vtensor<[2,3,5],f32>, !torch.vtensor<[2,3,1],f32>, !torch.vtensor<[2,3,1],f32>
  }
}

