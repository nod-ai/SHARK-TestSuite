// Transform dialect specification for attention on MI300 with MFMA.
// This script only supports variants of attention with a sequence
// length that is a multiple of 64. There are two near duplicate
// because we need different tile sizes when the head dimension is 512.
// TODO: Figure out how to parameterize the tile sizes without duplicating
// the attention function.

#layout_16 = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
#layout = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>

module attributes { transform.with_named_sequence } {
//===----------------------------------------------------------------------===//
// Attention
//===----------------------------------------------------------------------===//

  // Utility matching for finding all undistributed fills.
  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["linalg.fill"] : !transform.any_op
    %0 = transform.get_parent_op %arg0 {allow_empty_results, nth_parent = 2 : i64, op_name = "scf.forall"} : (!transform.any_op) -> !transform.any_op
    transform.match.operation_empty %0 : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @get_undistributed_fills(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield %0 : !transform.any_op
  }

  // Script for FA2 transform pipeline when head_dim % 64 = 0.
  transform.named_sequence @__attention_main(%variant_op: !transform.any_op {transform.readonly}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_attention, %forall_grid =
    transform.structured.tile_using_forall %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Tile batch dimensions of attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %top_level_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %top_level_func : !transform.any_op

    // Promote query and output operands
    // ==========================================
    //%attention3 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    //%promoted_attention, %alloc_a0, %alloc_a1 = transform.iree.promote_operands %attention3 [0, 3]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile and decompose attention
    // ==========================================
    %attention4 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill, %inner_loop, %final_scaling, %last_truncate, %blocked_attention = transform.iree.tile_attention %attention4 {tile_size = 32} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %scale_q, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %scale_factor, %update, %reduce_sum, %truncate, %scale_acc, %second_matmul
        = transform.iree.decompose_tiled_attention %blocked_attention {tile_size = 32} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Promote key and value operands
    // ==========================================
    %promoted_first_matmul, %alloc0 = transform.iree.promote_operands %first_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %promoted_second_matmul, %alloc1 = transform.iree.promote_operands %second_matmul [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile and fuse attention ops
    // ==========================================
    %tiled_matmul, %forall = transform.structured.tile_using_forall %promoted_second_matmul tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_reduce_sum, %forall_reduce = transform.structured.tile_using_forall %reduce_sum tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    %f0, %loop0 = transform.structured.fuse_into_containing_op %scale_acc into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f1, %loop1 = transform.structured.fuse_into_containing_op %truncate into %loop0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %loop4 = transform.loop.fuse_sibling %forall_reduce into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f5_1, %loop5_1 = transform.structured.fuse_into_containing_op %update into %loop4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f5, %loop5 = transform.structured.fuse_into_containing_op %scale_factor into %loop5_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f6, %loop6 = transform.structured.fuse_into_containing_op %partial_softmax into %loop5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f7, %loop7 = transform.structured.fuse_into_containing_op %reduce_max into %loop6 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f8, %loop8 = transform.structured.fuse_into_containing_op %promoted_first_matmul into %loop7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f9, %loop9 = transform.structured.fuse_into_containing_op %fill_op into %loop8 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f10, %loop10 = transform.structured.fuse_into_containing_op %scale_q into %loop9 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Distribute fills
    // ==========================================

    // Get all fills that haven't been distributed to warps.
    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    %tiled_fill, %fill_grid = transform.structured.tile_using_forall %fills tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Distribute last_truncate and fuse final_scaling into it
    // ==========================================
    %tiled_truncate, %loop_truncate = transform.structured.tile_using_forall %last_truncate tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %final_scaling into %loop_truncate : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    // Get the vector.contract ops.
    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %layout16x16x16 = transform.param.constant #layout -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract1, %layout16x16x16 { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %layout16x16x16 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.amdgpu_distribute_vectors %distribute_func : !transform.any_op

    %distribute_func_2 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    transform.iree.gpu_distribute_shared_memory_copy %distribute_func_2 : (!transform.any_op) -> ()
    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    transform.iree.reduce_shared_memory_bank_conflicts %distribute_func_2 : (!transform.any_op) -> ()

    transform.yield
  }

  // Script for FA2 transform pipeline for head_dim = 512.
  // For head_dim = 512, since the matmul is so big, and just try to do a single wave big load + big mfma.
  transform.named_sequence @__attention_main_len_512(%variant_op: !transform.any_op {transform.readonly}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %tiled_attention, %forall_grid =
    transform.structured.tile_using_forall %attention tile_sizes [1, 64]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Tile batch dimensions of attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %top_level_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %top_level_func : !transform.any_op

    // Promote query and output operands
    // ==========================================
    //%attention3 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    //%promoted_attention, %alloc_a0, %alloc_a1 = transform.iree.promote_operands %attention3 [0, 3]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Tile and decompose attention
    // ==========================================
    %attention4 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %acc_fill, %max_fill, %sum_fill, %inner_loop, %final_scaling, %last_truncate, %blocked_attention = transform.iree.tile_attention %attention4 {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %scale_q, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %scale_factor, %update, %reduce_sum, %truncate, %scale_acc, %second_matmul
        = transform.iree.decompose_tiled_attention %blocked_attention {tile_size = 64} :
      (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Promote key and value operands
    // ==========================================
    // %promoted_first_matmul, %alloc0 = transform.iree.promote_operands %first_matmul [1]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // %promoted_second_matmul, %alloc1 = transform.iree.promote_operands %second_matmul [1]
    //  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile and fuse attention ops
    // ==========================================
    %tiled_matmul, %forall = transform.structured.tile_using_forall %second_matmul tile_sizes [16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_reduce_sum, %forall_reduce = transform.structured.tile_using_forall %reduce_sum tile_sizes [16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    %f0, %loop0 = transform.structured.fuse_into_containing_op %scale_acc into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f1, %loop1 = transform.structured.fuse_into_containing_op %truncate into %loop0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %loop4 = transform.loop.fuse_sibling %forall_reduce into %loop1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f5_1, %loop5_1 = transform.structured.fuse_into_containing_op %update into %loop4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f5, %loop5 = transform.structured.fuse_into_containing_op %scale_factor into %loop5_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f6, %loop6 = transform.structured.fuse_into_containing_op %partial_softmax into %loop5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_cse to %func : !transform.any_op

    %f7, %loop7 = transform.structured.fuse_into_containing_op %reduce_max into %loop6 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %f8, %loop8 = transform.structured.fuse_into_containing_op %first_matmul into %loop7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f9, %loop9 = transform.structured.fuse_into_containing_op %fill_op into %loop8 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %f10, %loop10 = transform.structured.fuse_into_containing_op %scale_q into %loop9 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Distribute fills
    // ==========================================

    // Get all fills that haven't been distributed to warps.
    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    %tiled_fill, %fill_grid = transform.structured.tile_using_forall %fills tile_sizes[16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Distribute last_truncate and fuse final_scaling into it
    // ==========================================
    %tiled_truncate, %loop_truncate = transform.structured.tile_using_forall %last_truncate tile_sizes[16] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %final_scaling into %loop_truncate : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, 4, 1] subgroup_size = 64 : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    // transform.print %memref_func : !transform.any_op

    // Get the vector.contract ops.
    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %layout16x16x16 = transform.param.constant #layout_16 -> !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract1, %layout16x16x16 { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %layout16x16x16 : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.amdgpu_distribute_vectors %distribute_func : !transform.any_op

    %distribute_func_2 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    %func_10 = transform.structured.match ops{["func.func"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    transform.iree.gpu_distribute_shared_memory_copy %func_10 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func_10 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %func_10 : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %distribute_func_2 : (!transform.any_op) -> !transform.any_op
    transform.iree.reduce_shared_memory_bank_conflicts %func_11 : (!transform.any_op) -> ()

    transform.yield
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention_len_512(%attention: !transform.any_op {transform.readonly}) {
    %func = transform.get_parent_op %attention {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main_len_512, {"amdgpu-waves-per-eu" = 1}> -> !transform.any_param
    transform.annotate %func "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention_len_512(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x512xf16> : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_attention(%attention: !transform.any_op {transform.readonly}) {
    %func = transform.get_parent_op %attention {op_name = "func.func"} : (!transform.any_op) -> !transform.any_op
    %attn = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__attention_main, {"amdgpu-waves-per-eu" = 2}> -> !transform.any_param
    transform.annotate %func "translation_info" = %attn : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @match_attention(%attention: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %attention ["iree_linalg_ext.attention"] : !transform.any_op
    %in0 = transform.get_operand %attention[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?x?x?xf16> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[2], 64 : !transform.any_value
    transform.yield %attention : !transform.any_op
  }

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

  transform.named_sequence @__kernel_config(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        // Attention.
        @match_attention_len_512 -> @custom_attention_len_512,
        @match_attention -> @custom_attention
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} ////  module
