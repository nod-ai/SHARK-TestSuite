module {
  func.func @test_einsum_inner_prod(%arg0: !torch.vtensor<[5],f64>, %arg1: !torch.vtensor<[5],f64>) -> !torch.vtensor<[],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Einsum"(%arg0, %arg1) {torch.onnx.equation = "i,i"} : (!torch.vtensor<[5],f64>, !torch.vtensor<[5],f64>) -> !torch.vtensor<[],f64>
    return %0 : !torch.vtensor<[],f64>
  }
}

