module {
  func.func @test_sce_mean_3d_expanded(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[0, 0, -1]> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %1 = torch.operator "onnx.Reshape"(%arg0, %0) : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,5,2],f32> 
    %2 = torch.operator "onnx.Transpose"(%1) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[3,5,2],f32>) -> !torch.vtensor<[3,2,5],f32> 
    %3 = torch.operator "onnx.LogSoftmax"(%2) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[3,2,5],f32>) -> !torch.vtensor<[3,2,5],f32> 
    %4 = torch.operator "onnx.Transpose"(%3) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[3,2,5],f32>) -> !torch.vtensor<[3,5,2],f32> 
    %5 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[3,5,2],f32>) -> !torch.vtensor<[3],si64> 
    %6 = torch.operator "onnx.Reshape"(%4, %5) : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,5,2],f32> 
    %7 = call @"('NegativeLogLikelihoodLoss', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A], input: \22SoftmaxCrossEntropyLoss_test_sce_mean_3d_expanded_function_X_Log\22\0Ainput: \22y\22\0Aoutput: \22z\22\0Aop_type: \22NegativeLogLikelihoodLoss\22\0Aattribute {\0A  name: \22reduction\22\0A  s: \22mean\22\0A  type: STRING\0A}\0Adomain: \22\22\0A)"(%6, %arg1) : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32>
    return %7 : !torch.vtensor<[],f32>
  }
  func.func private @"('NegativeLogLikelihoodLoss', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A  }\0A}\0A], input: \22SoftmaxCrossEntropyLoss_test_sce_mean_3d_expanded_function_X_Log\22\0Ainput: \22y\22\0Aoutput: \22z\22\0Aop_type: \22NegativeLogLikelihoodLoss\22\0Aattribute {\0A  name: \22reduction\22\0A  s: \22mean\22\0A  type: STRING\0A}\0Adomain: \22\22\0A)"(%arg0: !torch.vtensor<[3,5,2],f32>, %arg1: !torch.vtensor<[3,2],si64>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %3 = torch.operator "onnx.Unsqueeze"(%arg1, %2) : (!torch.vtensor<[3,2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],si64> 
    %4 = torch.operator "onnx.GatherElements"(%arg0, %3) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,5,2],f32>, !torch.vtensor<[3,1,2],si64>) -> !torch.vtensor<[3,1,2],f32> 
    %5 = torch.operator "onnx.Neg"(%4) : (!torch.vtensor<[3,1,2],f32>) -> !torch.vtensor<[3,1,2],f32> 
    %6 = torch.operator "onnx.Slice"(%5, %0, %1, %1) : (!torch.vtensor<[3,1,2],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> 
    %7 = torch.operator "onnx.Squeeze"(%6, %2) : (!torch.vtensor<[3,1,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,2],f32> 
    %8 = torch.operator "onnx.ReduceMean"(%7) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,2],f32>) -> !torch.vtensor<[],f32> 
    return %8 : !torch.vtensor<[],f32>
  }
}

