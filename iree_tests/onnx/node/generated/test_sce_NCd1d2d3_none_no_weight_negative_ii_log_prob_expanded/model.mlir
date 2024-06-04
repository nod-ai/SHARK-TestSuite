module {
  func.func public @test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded(%arg0: !torch.vtensor<[3,5,6,6,5],f32>, %arg1: !torch.vtensor<[3,6,6,5],si64>) -> (!torch.vtensor<[3,6,6,5],f32>, !torch.vtensor<[3,5,6,6,5],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[0, 0, -1]> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %1 = torch.operator "onnx.Reshape"(%arg0, %0) : (!torch.vtensor<[3,5,6,6,5],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,5,180],f32> 
    %2 = torch.operator "onnx.Transpose"(%1) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[3,5,180],f32>) -> !torch.vtensor<[3,180,5],f32> 
    %3 = call @"('LogSoftmax', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 180\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 180\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded_function_X_NDC\22\0Aoutput: \22SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded_function_X_LogSM\22\0Aop_type: \22LogSoftmax\22\0Aattribute {\0A  name: \22axis\22\0A  i: 2\0A  type: INT\0A}\0Adomain: \22\22\0A)"(%2) : (!torch.vtensor<[3,180,5],f32>) -> !torch.vtensor<[3,180,5],f32>
    %4 = torch.operator "onnx.Transpose"(%3) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[3,180,5],f32>) -> !torch.vtensor<[3,5,180],f32> 
    %5 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[3,5,6,6,5],f32>) -> !torch.vtensor<[5],si64> 
    %6 = torch.operator "onnx.Reshape"(%4, %5) : (!torch.vtensor<[3,5,180],f32>, !torch.vtensor<[5],si64>) -> !torch.vtensor<[3,5,6,6,5],f32> 
    %7 = torch.operator "onnx.Identity"(%6) : (!torch.vtensor<[3,5,6,6,5],f32>) -> !torch.vtensor<[3,5,6,6,5],f32> 
    %8 = call @"('NegativeLogLikelihoodLoss', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded_function_X_Log\22\0Ainput: \22y\22\0Aoutput: \22z\22\0Aop_type: \22NegativeLogLikelihoodLoss\22\0Aattribute {\0A  name: \22reduction\22\0A  s: \22none\22\0A  type: STRING\0A}\0Aattribute {\0A  name: \22ignore_index\22\0A  i: -5\0A  type: INT\0A}\0Adomain: \22\22\0A)"(%6, %arg1) : (!torch.vtensor<[3,5,6,6,5],f32>, !torch.vtensor<[3,6,6,5],si64>) -> !torch.vtensor<[3,6,6,5],f32>
    return %8, %7 : !torch.vtensor<[3,6,6,5],f32>, !torch.vtensor<[3,5,6,6,5],f32>
  }
  func.func private @"('LogSoftmax', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 180\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 180\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded_function_X_NDC\22\0Aoutput: \22SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded_function_X_LogSM\22\0Aop_type: \22LogSoftmax\22\0Aattribute {\0A  name: \22axis\22\0A  i: 2\0A  type: INT\0A}\0Adomain: \22\22\0A)"(%arg0: !torch.vtensor<[3,180,5],f32>) -> !torch.vtensor<[3,180,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.ReduceMax"(%arg0) {torch.onnx.axes = [2 : si64], torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,180,5],f32>) -> !torch.vtensor<[3,180,1],f32> 
    %2 = torch.operator "onnx.Sub"(%arg0, %1) : (!torch.vtensor<[3,180,5],f32>, !torch.vtensor<[3,180,1],f32>) -> !torch.vtensor<[3,180,5],f32> 
    %3 = torch.operator "onnx.Exp"(%2) : (!torch.vtensor<[3,180,5],f32>) -> !torch.vtensor<[3,180,5],f32> 
    %4 = torch.operator "onnx.ReduceSum"(%3, %0) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,180,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,180,1],f32> 
    %5 = torch.operator "onnx.Log"(%4) : (!torch.vtensor<[3,180,1],f32>) -> !torch.vtensor<[3,180,1],f32> 
    %6 = torch.operator "onnx.Sub"(%2, %5) : (!torch.vtensor<[3,180,5],f32>, !torch.vtensor<[3,180,1],f32>) -> !torch.vtensor<[3,180,5],f32> 
    return %6 : !torch.vtensor<[3,180,5],f32>
  }
  func.func private @"('NegativeLogLikelihoodLoss', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 6\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22SoftmaxCrossEntropyLoss_test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob_expanded_function_X_Log\22\0Ainput: \22y\22\0Aoutput: \22z\22\0Aop_type: \22NegativeLogLikelihoodLoss\22\0Aattribute {\0A  name: \22reduction\22\0A  s: \22none\22\0A  type: STRING\0A}\0Aattribute {\0A  name: \22ignore_index\22\0A  i: -5\0A  type: INT\0A}\0Adomain: \22\22\0A)"(%arg0: !torch.vtensor<[3,5,6,6,5],f32>, %arg1: !torch.vtensor<[3,6,6,5],si64>) -> !torch.vtensor<[3,6,6,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %3 = torch.operator "onnx.Unsqueeze"(%arg1, %2) : (!torch.vtensor<[3,6,6,5],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6,5],si64> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-5> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %5 = torch.operator "onnx.Sub"(%3, %3) : (!torch.vtensor<[3,1,6,6,5],si64>, !torch.vtensor<[3,1,6,6,5],si64>) -> !torch.vtensor<[3,1,6,6,5],si64> 
    %6 = torch.operator "onnx.Cast"(%3) {torch.onnx.to = 7 : si64} : (!torch.vtensor<[3,1,6,6,5],si64>) -> !torch.vtensor<[3,1,6,6,5],si64> 
    %7 = torch.operator "onnx.Equal"(%6, %4) : (!torch.vtensor<[3,1,6,6,5],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6,5],i1> 
    %8 = torch.operator "onnx.Where"(%7, %5, %3) : (!torch.vtensor<[3,1,6,6,5],i1>, !torch.vtensor<[3,1,6,6,5],si64>, !torch.vtensor<[3,1,6,6,5],si64>) -> !torch.vtensor<[3,1,6,6,5],si64> 
    %9 = torch.operator "onnx.GatherElements"(%arg0, %8) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,5,6,6,5],f32>, !torch.vtensor<[3,1,6,6,5],si64>) -> !torch.vtensor<[3,1,6,6,5],f32> 
    %10 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.000000e+00> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %11 = torch.operator "onnx.Where"(%7, %10, %9) : (!torch.vtensor<[3,1,6,6,5],i1>, !torch.vtensor<[1],f32>, !torch.vtensor<[3,1,6,6,5],f32>) -> !torch.vtensor<[3,1,6,6,5],f32> 
    %12 = torch.operator "onnx.Neg"(%11) : (!torch.vtensor<[3,1,6,6,5],f32>) -> !torch.vtensor<[3,1,6,6,5],f32> 
    %13 = torch.operator "onnx.Slice"(%12, %0, %1, %1) : (!torch.vtensor<[3,1,6,6,5],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6,5],f32> 
    %14 = torch.operator "onnx.Squeeze"(%7, %2) : (!torch.vtensor<[3,1,6,6,5],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,6,6,5],i1> 
    %15 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %16 = torch.operator "onnx.Where"(%14, %10, %15) : (!torch.vtensor<[3,6,6,5],i1>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,6,6,5],f32> 
    %17 = torch.operator "onnx.Squeeze"(%13, %2) : (!torch.vtensor<[3,1,6,6,5],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,6,6,5],f32> 
    %18 = torch.operator "onnx.Mul"(%17, %16) : (!torch.vtensor<[3,6,6,5],f32>, !torch.vtensor<[3,6,6,5],f32>) -> !torch.vtensor<[3,6,6,5],f32> 
    return %18 : !torch.vtensor<[3,6,6,5],f32>
  }
}

