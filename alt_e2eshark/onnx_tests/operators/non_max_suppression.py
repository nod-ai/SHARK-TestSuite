# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors

class NonMaxSuppressionTestBasic(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 10, 4]),
            make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 10]),
            make_tensor_value_info("max_output_boxes_per_class", TensorProto.INT64, [1]),
            make_tensor_value_info("iou_threshold", TensorProto.FLOAT, [1]),
            make_tensor_value_info("score_threshold", TensorProto.FLOAT, [1]),
        ]
        # Create ValueInfoProto object for selected_indices tensor
        self.output_vi = [make_tensor_value_info("selected_indices", TensorProto.INT64, [1, 3])]

    def construct_nodes(self):
        node = make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
            center_point_box=0,
        )
        self.node_list.append(node)

    def construct_inputs(self):
            boxes = np.array(
                [
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ]
                ]).astype(np.float32)
            scores = np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([1]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            inputs = TestTensors((boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold))
            return inputs

register_test(NonMaxSuppressionTestBasic, "non_max_suppression_test_basic")

class NonMaxSuppressionTestCenterPointBox(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 6, 4]),
            make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 6]),
            make_tensor_value_info("max_output_boxes_per_class", TensorProto.INT64, [1]),
            make_tensor_value_info("iou_threshold", TensorProto.FLOAT, [1]),
            make_tensor_value_info("score_threshold", TensorProto.FLOAT, [1]),
        ]
        # Create ValueInfoProto object for selected_indices tensor
        self.output_vi = [make_tensor_value_info("selected_indices", TensorProto.INT64, [3, 3])]

    def construct_nodes(self):
        node = make_node(
            "NonMaxSuppression",
            inputs=[
                "boxes",
                "scores",
                "max_output_boxes_per_class",
                "iou_threshold",
                "score_threshold",
            ],
            outputs=["selected_indices"],
            center_point_box=1,
        )
        self.node_list.append(node)

    def construct_inputs(self):
            boxes = np.array(
                [
                    [
                        [0.5, 0.5, 1.0, 1.0],
                        [0.5, 0.6, 1.0, 1.0],
                        [0.5, 0.4, 1.0, 1.0],
                        [0.5, 10.5, 1.0, 1.0],
                        [0.5, 10.6, 1.0, 1.0],
                        [0.5, 100.5, 1.0, 1.0],
                    ]
                ]).astype(np.float32)
            scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
            max_output_boxes_per_class = np.array([1]).astype(np.int64)
            iou_threshold = np.array([0.5]).astype(np.float32)
            score_threshold = np.array([0.0]).astype(np.float32)
            inputs = TestTensors((boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold))
            return inputs

register_test(NonMaxSuppressionTestCenterPointBox, "non_max_suppression_test_center_point_box")