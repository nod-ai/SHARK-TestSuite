module {
  func.func public @test_dynamicquantizelinear_expanded(%arg0: !torch.vtensor<[6],f32>) -> (!torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2.550000e+02> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.ReduceMin"(%arg0) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.Min"(%2, %0) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.ReduceMax"(%arg0) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[6],f32>) -> !torch.vtensor<[],f32> 
    %5 = torch.operator "onnx.Max"(%4, %0) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Sub"(%5, %3) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %7 = torch.operator "onnx.Div"(%6, %1) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %8 = torch.operator "onnx.Div"(%3, %7) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %9 = torch.operator "onnx.Sub"(%0, %8) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %10 = call @"('Clip', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A], input: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Initial_ZeroPoint_FP\22\0Ainput: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Q_Min\22\0Ainput: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Q_Max\22\0Aoutput: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Clipped_ZeroPoint_FP\22\0Aop_type: \22Clip\22\0Adomain: \22\22\0A)"(%9, %0, %1) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32>
    %11 = torch.operator "onnx.Round"(%10) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %12 = torch.operator "onnx.Cast"(%11) {torch.onnx.to = 2 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],ui8> 
    %13 = torch.operator "onnx.Identity"(%7) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %14 = torch.operator "onnx.Identity"(%12) : (!torch.vtensor<[],ui8>) -> !torch.vtensor<[],ui8> 
    %15 = torch.operator "onnx.QuantizeLinear"(%arg0, %7, %12) : (!torch.vtensor<[6],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[6],ui8> 
    return %15, %13, %14 : !torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>
  }
  func.func private @"('Clip', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A, tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A], input: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Initial_ZeroPoint_FP\22\0Ainput: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Q_Min\22\0Ainput: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Q_Max\22\0Aoutput: \22DynamicQuantizeLinear_test_dynamicquantizelinear_expanded_function_Clipped_ZeroPoint_FP\22\0Aop_type: \22Clip\22\0Adomain: \22\22\0A)"(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Less"(%arg0, %arg1) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],i1> 
    %1 = torch.operator "onnx.Where"(%0, %arg1, %arg0) : (!torch.vtensor<[],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Less"(%arg2, %1) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],i1> 
    %3 = torch.operator "onnx.Where"(%2, %arg2, %1) : (!torch.vtensor<[],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    return %3 : !torch.vtensor<[],f32>
  }
}

