// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/genai/perf_metrics.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>

#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "stateful_pipeline_base.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

/**
 * @brief Target model wrapper for DFlash speculation
 *
 * Handles target model inference and hidden state extraction.
 * Reuses Eagle3InferWrapperBase for KV cache management and sampling.
 */
class DFlashTargetWrapper {
public:
    explicit DFlashTargetWrapper(const ov::genai::ModelDesc& model_desc);
    ~DFlashTargetWrapper() = default;

    void initialize_sequence(const ov::Tensor& input_ids, const ov::genai::GenerationConfig& config);
    void append_tokens(const std::vector<int64_t>& tokens);
    void truncate_sequence(size_t size);
    void trim_kv_cache(size_t tokens_to_remove);
    void save_linear_states();
    void restore_linear_states();
    void reset_state();
    void release_memory();

    size_t get_sequence_length() const;
    const std::vector<int64_t>& get_generated_tokens() const;
    SequenceGroup::Ptr get_sequence_group() const { return m_sequence_group; }
    Sequence::Ptr get_current_sequence() const;

    /// @brief Runs target model inference and returns logits + hidden features
    struct TargetOutput {
        ov::Tensor logits;
        ov::Tensor hidden_features;
    };
    TargetOutput infer(size_t input_token_count);

    /// @brief Sample from full validation logits
    /// @param num_tokens_to_validate Number of draft tokens to validate
    /// @return Validated tokens (accepted draft + bonus)
    std::vector<int64_t> sample_validate(const ov::Tensor& logits,
                                         size_t input_token_count,
                                         size_t sample_count,
                                         size_t num_tokens_to_validate);

    /// @brief Sample single token (for prefill)
    std::vector<int64_t> sample_single(const ov::Tensor& logits, size_t input_token_count);

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() { return m_raw_perf_metrics; }

private:
    void build_model_inputs(size_t input_token_count,
                            ov::Tensor& input_ids,
                            ov::Tensor& attention_mask,
                            ov::Tensor& position_ids);
    uint64_t execute_inference();

    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;

    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    ov::genai::utils::CacheTypes m_cache_types;
    size_t m_position_ids_rank = 2;  // Default: [batch, seq_len]. MROPE models use 3: [num_heads, batch, seq_len]
    size_t m_position_ids_leading_dim = 1;  // For MROPE: e.g. 4
    std::vector<std::pair<std::string, ov::Tensor>> m_saved_linear_states;
};

/**
 * @brief Draft model wrapper for DFlash
 *
 * Executes a single forward pass to produce block_size logits.
 * The draft model takes:
 *   - input_ids [1, block_size] (anchor + MASK tokens)
 *   - target_hidden [1, ctx_len, N*H] (concatenated hidden states from target layers)
 * And produces:
 *   - logits [1, block_size, vocab_size]
 */
class DFlashDraftWrapper {
public:
    explicit DFlashDraftWrapper(const ov::genai::ModelDesc& model_desc,
                                const utils::dflash::DFlashRTInfo& config);
    ~DFlashDraftWrapper() = default;

    /// @brief Single forward pass producing block_size logits
    /// @param input_ids Token IDs [1, block_size] (anchor + MASKs)
    /// @param target_hidden Concatenated hidden states [1, ctx_len, N*H]
    /// @return Logits [1, block_size, vocab_size]
    ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& target_hidden);

    void trim_kv_cache(size_t tokens_to_remove);
    void reset_state();
    void release_memory();

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() { return m_raw_perf_metrics; }

private:
    uint64_t execute_inference();

    std::string m_device;
    ov::AnyMap m_properties;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    ov::genai::utils::CacheTypes m_cache_types;

    int m_block_size;
    int m_mask_token_id;

    ov::genai::RawPerfMetrics m_raw_perf_metrics;
};

/**
 * @brief Stateful DFlash speculative decoding pipeline
 *
 * Implements DFlash algorithm: draft model generates block_size tokens in a single call
 * using block diffusion (bidirectional attention), then target model validates them.
 *
 * Key differences from Eagle3:
 * - Draft produces all candidates in ONE forward pass (not iterative)
 * - Draft uses bidirectional attention (is_causal=False)
 * - Hidden state extraction uses N layers (5-6) instead of 3
 * - No FC move from draft to main (FC stays in draft)
 * - lm_head is grafted from target (tie_word_embeddings=true)
 */
class StatefulDFlashLLMPipeline : public StatefulSpeculativePipelineBase {
public:
    StatefulDFlashLLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                              const ov::genai::ModelDesc& draft_model_desc,
                              const utils::dflash::DFlashRTInfo& dflash_config);
    ~StatefulDFlashLLMPipeline();

    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;

    void finish_chat() override;

protected:
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;

    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    struct SpeculativeResult {
        size_t accepted_tokens_count = 0;
        bool eos_reached = false;
        std::vector<int64_t> validated_tokens;
    };

    SpeculativeResult run_speculative_iteration(int64_t eos_token_id,
                                                size_t current_generated_tokens,
                                                size_t max_new_tokens);

    std::unique_ptr<DFlashTargetWrapper> m_target;
    std::unique_ptr<DFlashDraftWrapper> m_draft;

    utils::dflash::DFlashRTInfo m_dflash_config;
    size_t m_prompt_length = 0;
    int64_t m_last_accepted_token = -1;
};

}  // namespace genai
}  // namespace ov
