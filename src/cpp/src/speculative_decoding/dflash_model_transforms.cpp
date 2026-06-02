// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <fstream>
#include <functional>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "eagle3_model_transforms.hpp"
#include "json_utils.hpp"
#include "logger.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/result.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    DFlashRTInfo info;

    // Check if dflash_mode is explicitly set in properties
    if (config.find("dflash_mode") != config.end()) {
        info.dflash_mode = config.at("dflash_mode").as<bool>();
        config.erase("dflash_mode");
    }

    // If not explicitly set, try to detect from config.json
    if (!info.dflash_mode && !models_path.empty()) {
        auto config_file_path = models_path / "config.json";
        if (std::filesystem::exists(config_file_path)) {
            std::ifstream file(config_file_path);
            nlohmann::json data = nlohmann::json::parse(file);

            // Check architectures for DFlashDraftModel
            if (data.contains("architectures") && data["architectures"].is_array()) {
                for (const auto& arch : data["architectures"]) {
                    auto arch_str = arch.get<std::string>();
                    if (arch_str == "DFlashDraftModel" || arch_str == "DFlashForExport") {
                        info.dflash_mode = true;
                        break;
                    }
                }
            }

            if (info.dflash_mode) {
                using ov::genai::utils::read_json_param;

                read_json_param(data, "block_size", info.block_size);
                read_json_param(data, "tie_word_embeddings", info.tie_word_embeddings);

                // Read dflash_config nested fields
                read_json_param(data, "dflash_config.mask_token_id", info.mask_token_id);
                read_json_param(data, "dflash_config.target_layer_ids", info.target_layer_ids);
            }
        }
    }

    // Also read from properties if provided
    if (config.find("block_size") != config.end()) {
        info.block_size = config.at("block_size").as<int>();
        config.erase("block_size");
    }
    if (config.find("mask_token_id") != config.end()) {
        info.mask_token_id = config.at("mask_token_id").as<int>();
        config.erase("mask_token_id");
    }
    if (config.find("target_layer_ids") != config.end()) {
        info.target_layer_ids = config.at("target_layer_ids").as<std::vector<int32_t>>();
        config.erase("target_layer_ids");
    }

    if (info.dflash_mode) {
        OPENVINO_ASSERT(!info.target_layer_ids.empty(),
                        "DFlash mode requires target_layer_ids to be specified in config.json or properties");
        OPENVINO_ASSERT(info.mask_token_id >= 0,
                        "DFlash mode requires mask_token_id to be specified in config.json or properties");
        OPENVINO_ASSERT(info.block_size > 0,
                        "DFlash block_size must be positive, got: ", info.block_size);
    }

    return info;
}

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (model->has_rt_info("dflash_mode") && model->get_rt_info<bool>("dflash_mode")) {
        properties["dflash_mode"] = true;
    }
}

void share_vocabulary_and_lm_head(const std::shared_ptr<ov::Model>& main_model,
                                   std::shared_ptr<ov::Model>& draft_model) {
    // The draft model was exported with a dummy (all-zeros) embedding weight.
    // We share the target model's real embed_tokens into the draft so it gets proper embeddings.

    // Remember the draft's embedding output type before sharing (target may differ in precision)
    ov::element::Type draft_embed_et;
    std::shared_ptr<ov::Node> draft_gather_node;
    for (const auto& node : draft_model->get_ordered_ops()) {
        auto gather = std::dynamic_pointer_cast<ov::op::util::GatherBase>(node);
        if (!gather) continue;
        auto data_node = gather->input_value(0).get_node_shared_ptr();
        ov::PartialShape ps = data_node->get_output_partial_shape(0);
        if (ps.rank().is_static() && ps.rank().get_length() >= 2 &&
            ps[0].is_static() && ps[0].get_length() > 1000) {
            draft_embed_et = gather->get_output_element_type(0);
            draft_gather_node = gather;
            break;
        }
    }

    eagle3::share_vocabulary(main_model, draft_model);

    // If the shared embedding now has a different precision, insert a Convert
    if (draft_gather_node) {
        // Force type recomputation after weight replacement
        draft_gather_node->validate_and_infer_types();
        auto new_et = draft_gather_node->get_output_element_type(0);
        if (new_et != draft_embed_et) {
            auto convert = std::make_shared<ov::op::v0::Convert>(
                draft_gather_node->output(0), draft_embed_et);
            convert->set_friendly_name("dflash_embed_type_convert");
            // Collect consumers first to avoid invalidating the iterator
            std::vector<ov::Input<ov::Node>> consumers;
            for (auto& target_input : draft_gather_node->output(0).get_target_inputs()) {
                if (target_input.get_node() != convert.get()) {
                    consumers.push_back(target_input);
                }
            }
            for (auto& target_input : consumers) {
                target_input.replace_source_output(convert->output(0));
            }
        }
    }

    // Find the lm_head weight from the MAIN model.
    std::shared_ptr<ov::Node> lm_head_weight_node;
    for (const auto& result : main_model->get_results()) {
        auto source = result->input_value(0).get_node_shared_ptr();
        if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(source)) {
            lm_head_weight_node = matmul->input_value(1).get_node_shared_ptr();
            break;
        }
    }

    if (!lm_head_weight_node) {
        std::cerr << "[DFlash DEBUG] Could not find lm_head weight in main model." << std::endl;
        return;
    }

    // Clone the lm_head weight subgraph into draft model (deep copy to avoid cross-model refs)
    std::function<std::shared_ptr<ov::Node>(const std::shared_ptr<ov::Node>&,
                                            std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>&)>
        clone_node_recursive =
            [&](const std::shared_ptr<ov::Node>& node,
                std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>& cloned_nodes) -> std::shared_ptr<ov::Node> {
        auto it = cloned_nodes.find(node.get());
        if (it != cloned_nodes.end()) return it->second;
        std::shared_ptr<ov::Node> cloned;
        if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            cloned = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(),
                                                            constant->get_shape(),
                                                            constant->get_data_ptr());
        } else {
            ov::OutputVector cloned_inputs;
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                auto input_node = node->get_input_node_shared_ptr(i);
                auto cloned_input = clone_node_recursive(input_node, cloned_nodes);
                cloned_inputs.push_back(cloned_input->output(node->get_input_source_output(i).get_index()));
            }
            cloned = node->clone_with_new_inputs(cloned_inputs);
        }
        cloned->set_friendly_name(node->get_friendly_name() + "_lm_head_for_draft");
        cloned_nodes[node.get()] = cloned;
        return cloned;
    };
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> cloned_nodes;
    auto cloned_lm_head_weight = clone_node_recursive(lm_head_weight_node, cloned_nodes);

    std::cerr << "[DFlash DEBUG] Found lm_head weight from main model: "
              << lm_head_weight_node->get_friendly_name()
              << " shape=" << lm_head_weight_node->get_output_partial_shape(0)
              << " type=" << lm_head_weight_node->get_output_element_type(0).get_type_name() << std::endl;

    // Step 3: Add lm_head MatMul at the draft model's output
    // Find the last Result node that outputs hidden_states (not a constant)
    for (const auto& result : draft_model->get_results()) {
        auto input_node = result->input_value(0).get_node_shared_ptr();
        // Skip constant results (like d2t mapping tables)
        if (ov::as_type_ptr<ov::op::v0::Constant>(input_node)) continue;

        // Ensure weight has same type as hidden_states for the MatMul
        auto hidden_et = input_node->get_output_element_type(0);
        auto weight_et = cloned_lm_head_weight->get_output_element_type(0);
        std::shared_ptr<ov::Node> weight_for_matmul = cloned_lm_head_weight;
        if (hidden_et != weight_et) {
            auto convert = std::make_shared<ov::op::v0::Convert>(cloned_lm_head_weight, hidden_et);
            convert->set_friendly_name("dflash_lm_head_weight_convert");
            weight_for_matmul = convert;
        }

        // logits = MatMul(hidden_states, lm_head_weight^T)
        // MatMul with transpose_b=true: [B, seq, H] x [vocab, H]^T -> [B, seq, vocab]
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input_node, weight_for_matmul, false, true);
        matmul->set_friendly_name("dflash_lm_head");
        result->input(0).replace_source_output(matmul);

        // Name the output tensor "logits" so the runtime can find it by name
        result->output(0).set_names({"logits"});

        std::cerr << "[DFlash DEBUG] Grafted lm_head MatMul: hidden_et=" << hidden_et.get_type_name()
                  << " weight_et=" << weight_et.get_type_name()
                  << " weight_shape=" << cloned_lm_head_weight->get_output_partial_shape(0)
                  << " result_name=" << result->get_friendly_name() << std::endl;
        break;
    }
}

void transform_target_hidden_state(std::shared_ptr<ov::Model>& model,
                                    const std::vector<int32_t>& target_layer_ids) {
    // Reuse the generalized transform_hidden_state from eagle3 namespace
    eagle3::transform_hidden_state(model, target_layer_ids);
}

void ensure_target_has_input_ids(std::shared_ptr<ov::Model>& model) {
    // Check if model already has input_ids — if so, nothing to do
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == "input_ids") {
            return;
        }
    }

    // Find the inputs_embeds parameter
    std::shared_ptr<ov::op::v0::Parameter> embeds_param;
    for (const auto& param : model->get_parameters()) {
        if (param->get_friendly_name() == "inputs_embeds") {
            embeds_param = param;
            break;
        }
    }
    if (!embeds_param) {
        return;  // Model doesn't have inputs_embeds either — unusual, skip
    }

    // Find the lm_head weight (assumed to be the embedding weight for tied models).
    // Strategy: trace backward from the logits Result to find the final MatMul weight.
    std::shared_ptr<ov::Node> embed_weight_node;
    for (const auto& result : model->get_results()) {
        auto source = result->input_value(0).get_node_shared_ptr();
        if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(source)) {
            // Weight is input[1] of the lm_head MatMul
            embed_weight_node = matmul->input_value(1).get_node_shared_ptr();
            break;
        }
    }

    if (!embed_weight_node) {
        GENAI_INFO("DFlash: Could not find lm_head weight in target model, cannot add embedding.");
        return;
    }

    GENAI_INFO("DFlash: Target model has 'inputs_embeds' but no 'input_ids'. "
               "Adding embedding lookup from lm_head weight (tie_word_embeddings).");

    // Create new input_ids Parameter: [batch, seq_len], i64
    auto input_ids_param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, -1});
    input_ids_param->set_friendly_name("input_ids");
    input_ids_param->output(0).set_names({"input_ids"});

    // Create Gather: embed_weight[input_ids] along axis 0
    // embed_weight shape: [vocab_size, hidden_size]
    auto axis_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, 0);
    auto gather = std::make_shared<ov::op::v8::Gather>(embed_weight_node, input_ids_param, axis_const);
    gather->set_friendly_name("dflash_embed_tokens");

    // The Gather output may need a Convert if inputs_embeds was f32 and weight is bf16/f16
    auto gather_output = gather->output(0);
    auto embeds_type = embeds_param->get_element_type();
    auto weight_type = embed_weight_node->get_output_element_type(0);
    if (weight_type != embeds_type) {
        auto convert = std::make_shared<ov::op::v0::Convert>(gather_output, embeds_type);
        convert->set_friendly_name("dflash_embed_tokens_convert");
        gather_output = convert->output(0);
    }

    // Replace all consumers of inputs_embeds with the Gather output
    embeds_param->output(0).replace(gather_output);

    // Update model: remove inputs_embeds param, add input_ids param
    model->remove_parameter(embeds_param);
    model->add_parameters({input_ids_param});
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
