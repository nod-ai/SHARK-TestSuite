module {
  func.func @test_castlike_FLOAT_to_FLOAT8E4M3FNUZ(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],f8E5M2FNUZ>) -> !torch.vtensor<[3,4],f8E5M2FNUZ> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.CastLike"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],f8E5M2FNUZ>) -> !torch.vtensor<[3,4],f8E5M2FNUZ> 
    return %0 : !torch.vtensor<[3,4],f8E5M2FNUZ>
  }
}
