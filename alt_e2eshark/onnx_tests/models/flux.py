import os
import torch
from diffusers import FluxTransformer2DModel
from e2e_testing.framework import OnnxModelInfo, ImporterOptions
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors
from pathlib import Path

class FluxTransformerModelInfo(OnnxModelInfo):
    def __init__(self, name, onnx_model_path, opset_version = None):
        super().__init__(name, onnx_model_path, opset_version)
        self.hf_model_path = "black-forest-labs/FLUX.1-dev"
        self.model_dir = "transformer"
        self.img_height=1024
        self.img_width=1024
        self.compression_factor=8
        self.max_len=512
        self.bs=1
        self.torch_dtype=torch.float32
        self.param_path = str(Path(self.model).parent / "model_params.irpa")

    def update_dim_param_dict(self):
        self.dim_param_dict = {"B" : 1, "latent_dim" : 128, "L" : 128}

    def update_importer_options(self):
        self.importer_options = ImporterOptions(externalize_inputs_threshold=7, externalize_params=True, large_model=True)
        self.runtime_options = f" --parameters=model={self.param_path} --parameters=model={Path(self.model).parent / 'model.torch_onnx_params.irpa'}"

    def construct_inputs(self):
        latent_h, latent_w = self.img_height // self.compression_factor, self.img_width // self.compression_factor
        config = FluxTransformer2DModel.load_config(self.hf_model_path,
                                                        subfolder=self.model_dir)
        return TestTensors((torch.randn(self.bs, (latent_h // 2) * (latent_w // 2),
                                        config["in_channels"],
                                        dtype=self.torch_dtype),
                            torch.randn(self.bs,
                                        self.max_len,
                                        config['joint_attention_dim'],
                                        dtype=self.torch_dtype),
                            torch.randn(self.bs,
                                        config['pooled_projection_dim'],
                                        dtype=self.torch_dtype),
                            torch.tensor([1.]*self.bs, dtype=self.torch_dtype),
                            torch.randn((latent_h // 2) * (latent_w // 2),
                                        3,
                                        dtype=self.torch_dtype),
                            torch.randn(self.max_len, 3, dtype=self.torch_dtype),
                            torch.tensor([1.]*self.bs, dtype=self.torch_dtype),))

    def construct_model(self):
        input_names = [
            'hidden_states', 'encoder_hidden_states', 'pooled_projections',
            'timestep', 'img_ids', 'txt_ids', 'guidance'
        ]
        if not os.path.isfile(self.model):
            print("loading model")
            model = FluxTransformer2DModel.from_pretrained(self.hf_model_path,
                                                            subfolder=self.model_dir,
                                                            torch_dtype=self.torch_dtype)
            print("model loaded")
            from iree.turbine.aot import save_module_parameters
            print("saving params")
            save_module_parameters(self.param_path, model)
            print("params saved")

            output_names = ["latent"]
            dynamic_axes = {
                'hidden_states': {0: 'B', 1: 'latent_dim'},
                'encoder_hidden_states': {0: 'B',1: 'L'},
                'pooled_projections': {0: 'B'},
                'timestep': {0: 'B'},
                'img_ids': {0: 'latent_dim'},
                'txt_ids': {0: 'L'},
                'guidance': {0: 'B'},
            }
            sample_inputs= self.construct_inputs().data
            with torch.inference_mode():
                print("exporting hf model to onnx")
                torch.onnx.export(model,
                                    sample_inputs,
                                    self.model,
                                    export_params=False,
                                    do_constant_folding=False,
                                    input_names=input_names,
                                    output_names=output_names,
                                    dynamic_axes=dynamic_axes)
                print("onnx model generated")

        if not os.path.isfile(self.model):
            raise RuntimeError(f"Torch onnx export failed to produce an onnx model at {self.model}")
    
    def forward(self, input):
        with torch.inference_mode():
            model = FluxTransformer2DModel.from_pretrained(self.hf_model_path,
                                                            subfolder=self.model_dir,
                                                            torch_dtype=self.torch_dtype)
            data = input.to_torch().data
            return TestTensors(model.forward(data[0], data[1],data[2],data[3],data[4],data[5],data[6], return_dict=False))



register_test(FluxTransformerModelInfo, "flux_1_dev_transformer")



