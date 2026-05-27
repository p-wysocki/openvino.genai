// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

/**
 * @brief Runtime configuration for DFlash speculative decoding.
 */
struct DFlashRTInfo {
    bool dflash_mode = false;                      ///< Enable DFlash mode
    int block_size = 16;                           ///< Number of tokens produced per draft call
    int mask_token_id = -1;                        ///< Token ID for MASK fill in block input
    std::vector<int32_t> target_layer_ids;         ///< Target model layer indices to extract hidden states from
    bool tie_word_embeddings = true;               ///< Whether lm_head = embed_tokens^T
};

/**
 * @brief Extracts DFlash configuration from draft model config.
 *
 * Reads config.json from the draft model directory and checks for DFlash indicators:
 * - architectures contains "DFlashDraftModel"
 * - dflash_config section with mask_token_id, target_layer_ids
 * - block_size, tie_word_embeddings
 *
 * Also checks the properties map for "dflash_mode" flag set by apply_dflash_rt_info.
 *
 * @param config Model configuration map (may be modified to remove consumed keys).
 * @param models_path Path to draft model directory for reading config.json.
 * @return DFlashRTInfo structure with extracted configuration.
 */
DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path = {});

/**
 * @brief Applies DFlash runtime info from model to properties map.
 * @param model Model containing rt_info fields.
 * @param properties Properties map to update.
 */
void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

/**
 * @brief Shares embedding weights and grafts lm_head into draft model.
 *
 * For DFlash models with tie_word_embeddings=true:
 * 1. Copies embedding Gather from main model into draft model (like Eagle3)
 * 2. Adds a MatMul(hidden_states, embed_weight^T) at draft output to produce logits
 *
 * After this transform, the draft model produces logits directly.
 *
 * @param main_model Main (target) model to source embedding weights from.
 * @param draft_model Draft model to modify.
 */
void share_vocabulary_and_lm_head(const std::shared_ptr<ov::Model>& main_model,
                                   std::shared_ptr<ov::Model>& draft_model);

/**
 * @brief Extracts hidden states from specified target model layers for DFlash.
 *
 * Reuses the generalized transform_hidden_state from eagle3 namespace,
 * supporting any number of target layers (typically 5-6 for DFlash models).
 *
 * @param model Target model to transform.
 * @param target_layer_ids Layer indices to extract hidden states from.
 */
void transform_target_hidden_state(std::shared_ptr<ov::Model>& model,
                                    const std::vector<int32_t>& target_layer_ids);

/**
 * @brief Ensures the target model accepts input_ids instead of inputs_embeds.
 *
 * Models exported from VL pipelines (e.g., Qwen3.5) may have an `inputs_embeds`
 * parameter instead of `input_ids`. This transform adds an embedding lookup
 * (Gather) using the lm_head weight (which equals embed_tokens when
 * tie_word_embeddings=true), replacing `inputs_embeds` with `input_ids`.
 *
 * This is applied at load time so the user doesn't need to modify the IR.
 * If the model already has `input_ids`, this is a no-op.
 *
 * @param model Target model to transform (modified in-place).
 */
void ensure_target_has_input_ids(std::shared_ptr<ov::Model>& model);

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
