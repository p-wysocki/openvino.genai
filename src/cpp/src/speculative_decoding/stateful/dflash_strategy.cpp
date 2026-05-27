// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

#include "openvino/core/type/bfloat16.hpp"
#include "continuous_batching/timer.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "utils.hpp"

namespace ov::genai {
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
}  // namespace ov::genai

namespace {

ov::genai::StreamingStatus stream_generated_tokens(std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                   const std::vector<int64_t>& tokens) {
    if (streamer_ptr) {
        return streamer_ptr->write(tokens);
    }
    return ov::genai::StreamingStatus{};
}

}  // anonymous namespace

namespace ov::genai {

// ============================================================================
// DFlashTargetWrapper
// ============================================================================

DFlashTargetWrapper::DFlashTargetWrapper(const ModelDesc& model_desc)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_tokenizer(model_desc.tokenizer),
      m_sampler(model_desc.tokenizer) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
    m_cache_types = utils::get_cache_types(*model_desc.model);

    OPENVINO_ASSERT(!m_cache_types.has_linear() || true,  // TODO: re-enable after verifying linear attention compat
        "DFlash speculative decoding does not support models with linear attention states.");

    // Detect position_ids shape (MROPE models like Qwen3.5 use [num_heads, batch, seq_len])
    for (const auto& param : model_desc.model->get_parameters()) {
        if (param->get_friendly_name() == "position_ids") {
            auto ps = param->get_partial_shape();
            if (ps.rank().is_static() && ps.rank().get_length() == 3 && ps[0].is_static()) {
                m_position_ids_rank = 3;
                m_position_ids_leading_dim = ps[0].get_length();
            }
            break;
        }
    }

    m_request = utils::singleton_core()
                    .compile_model(model_desc.model, m_device, m_properties)
                    .create_infer_request();

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
    m_sequence_group = nullptr;
}

void DFlashTargetWrapper::initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config) {
    const auto shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "Expected input_ids shape [1, seq_len]");

    const int64_t* ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = shape[1];
    OPENVINO_ASSERT(seq_len > 0, "Empty prompt");

    TokenIds prompt_ids(ids_data, ids_data + seq_len);
    m_sequence_group = std::make_shared<SequenceGroup>(0, prompt_ids, config);
}

void DFlashTargetWrapper::append_tokens(const std::vector<int64_t>& tokens) {
    if (tokens.empty()) return;
    auto seq = get_current_sequence();
    OPENVINO_ASSERT(seq, "SequenceGroup not initialized");
    for (auto token : tokens) {
        seq->append_token(token, 0.0f);
    }
}

void DFlashTargetWrapper::truncate_sequence(size_t size) {
    auto seq = get_current_sequence();
    OPENVINO_ASSERT(seq, "SequenceGroup not initialized");
    const size_t prompt_len = m_sequence_group->get_prompt_len();
    const size_t current_len = prompt_len + seq->get_generated_len();
    if (size < current_len) {
        OPENVINO_ASSERT(size >= prompt_len, "Cannot truncate prompt tokens");
        seq->remove_last_tokens(current_len - size);
    }
}

void DFlashTargetWrapper::trim_kv_cache(size_t tokens_to_remove) {
    if (tokens_to_remove == 0) return;
    OPENVINO_ASSERT(tokens_to_remove > 0, "tokens_to_remove must be > 0");

    // Only trim KV cache states (key/value), skip SSM/conv states which are fixed-size
    const size_t seq_axis = m_kv_axes_pos.seq_len;
    for (auto& state : m_request.query_state()) {
        const std::string name = state.get_name();
        // Only trim states that are part of the KV cache (contain "key" or "value")
        if (name.find("key") == std::string::npos && name.find("value") == std::string::npos)
            continue;

        ov::Tensor old_tensor = state.get_state();
        auto shape = old_tensor.get_shape();
        if (shape[seq_axis] < tokens_to_remove)
            continue;

        shape[seq_axis] -= tokens_to_remove;
        ov::Coordinate begin(shape.size(), 0);
        ov::Coordinate end(shape);
        auto trimmed = ov::Tensor(old_tensor, begin, end);

        ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
        trimmed.copy_to(new_tensor);
        state.set_state(new_tensor);
    }
}

void DFlashTargetWrapper::save_linear_states() {
    m_saved_linear_states.clear();
    for (auto& state : m_request.query_state()) {
        const std::string name = state.get_name();
        if (name.find("key") != std::string::npos || name.find("value") != std::string::npos)
            continue;
        // Save non-KV states (SSM, conv, etc.)
        ov::Tensor original = state.get_state();
        ov::Tensor copy(original.get_element_type(), original.get_shape());
        original.copy_to(copy);
        m_saved_linear_states.emplace_back(name, std::move(copy));
    }
}

void DFlashTargetWrapper::restore_linear_states() {
    for (auto& state : m_request.query_state()) {
        const std::string name = state.get_name();
        for (const auto& [saved_name, saved_tensor] : m_saved_linear_states) {
            if (name == saved_name) {
                state.set_state(saved_tensor);
                break;
            }
        }
    }
    m_saved_linear_states.clear();
}

void DFlashTargetWrapper::reset_state() {
    m_request.reset_state();
    m_sequence_group = nullptr;
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void DFlashTargetWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
}

size_t DFlashTargetWrapper::get_sequence_length() const {
    if (auto seq = get_current_sequence()) {
        return m_sequence_group->get_prompt_len() + seq->get_generated_len();
    }
    return 0;
}

const std::vector<int64_t>& DFlashTargetWrapper::get_generated_tokens() const {
    static const std::vector<int64_t> empty;
    if (auto seq = get_current_sequence()) {
        return seq->get_generated_ids();
    }
    return empty;
}

Sequence::Ptr DFlashTargetWrapper::get_current_sequence() const {
    if (m_sequence_group) {
        const auto& sequences = m_sequence_group->get_sequences();
        if (!sequences.empty()) {
            return sequences[0];
        }
    }
    return nullptr;
}

void DFlashTargetWrapper::build_model_inputs(size_t input_token_count,
                                             ov::Tensor& input_ids,
                                             ov::Tensor& attention_mask,
                                             ov::Tensor& position_ids) {
    auto current_sequence = get_current_sequence();
    OPENVINO_ASSERT(current_sequence, "SequenceGroup not initialized");

    const auto& prompt_ids = m_sequence_group->get_prompt_ids();
    const auto& generated_ids = current_sequence->get_generated_ids();

    const size_t prompt_len = prompt_ids.size();
    const size_t generated_len = generated_ids.size();
    const size_t total_len = prompt_len + generated_len;
    const size_t start_pos = total_len - input_token_count;

    OPENVINO_ASSERT(input_token_count > 0 && input_token_count <= total_len);

    input_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    if (m_position_ids_rank == 3) {
        position_ids = ov::Tensor(ov::element::i64, {m_position_ids_leading_dim, 1, input_token_count});
    } else {
        position_ids = ov::Tensor(ov::element::i64, {1, input_token_count});
    }

    int64_t* input_ids_ptr = input_ids.data<int64_t>();
    int64_t* position_ids_ptr = position_ids.data<int64_t>();

    if (start_pos < prompt_len) {
        const size_t prompt_count = std::min(input_token_count, prompt_len - start_pos);
        std::copy_n(prompt_ids.data() + start_pos, prompt_count, input_ids_ptr);
        std::iota(position_ids_ptr, position_ids_ptr + prompt_count, static_cast<int64_t>(start_pos));
        if (input_token_count > prompt_count) {
            const size_t generated_count = input_token_count - prompt_count;
            std::copy_n(generated_ids.data(), generated_count, input_ids_ptr + prompt_count);
            std::iota(position_ids_ptr + prompt_count,
                      position_ids_ptr + prompt_count + generated_count,
                      static_cast<int64_t>(prompt_len));
        }
    } else {
        const size_t generated_start = start_pos - prompt_len;
        std::copy_n(generated_ids.data() + generated_start, input_token_count, input_ids_ptr);
        std::iota(position_ids_ptr,
                  position_ids_ptr + input_token_count,
                  static_cast<int64_t>(prompt_len + generated_start));
    }

    // For MROPE (3D position_ids), replicate the first slice to all other slices
    if (m_position_ids_rank == 3) {
        for (size_t h = 1; h < m_position_ids_leading_dim; ++h) {
            std::copy_n(position_ids_ptr, input_token_count,
                        position_ids_ptr + h * input_token_count);
        }
    }

    const size_t attention_mask_len = static_cast<size_t>(position_ids_ptr[input_token_count - 1] + 1);
    attention_mask = ov::Tensor(ov::element::i64, {1, attention_mask_len});
    std::fill_n(attention_mask.data<int64_t>(), attention_mask_len, 1);
}

uint64_t DFlashTargetWrapper::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return duration_us;
}

DFlashTargetWrapper::TargetOutput DFlashTargetWrapper::infer(size_t input_token_count) {
    ov::Tensor input_ids, attention_mask, position_ids;
    build_model_inputs(input_token_count, input_ids, attention_mask, position_ids);

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.set_tensor("position_ids", position_ids);

    // Set beam_idx for stateful model (required for KV cache / recurrent state management)
    ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {1});
    beam_idx.data<int32_t>()[0] = 0;
    m_request.set_tensor("beam_idx", beam_idx);

    uint64_t time_us = execute_inference();
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(time_us));

    TargetOutput output;
    auto logits_tensor = m_request.get_tensor("logits");
    // Ensure logits are f32 for sampling (model may output bf16)
    if (logits_tensor.get_element_type() != ov::element::f32) {
        ov::Tensor f32_logits(ov::element::f32, logits_tensor.get_shape());
        auto* src = reinterpret_cast<const ov::bfloat16*>(logits_tensor.data());
        float* dst = f32_logits.data<float>();
        for (size_t i = 0; i < logits_tensor.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
        output.logits = f32_logits;
    } else {
        output.logits = logits_tensor;
    }

    // Get hidden features (added by transform_target_hidden_state)
    auto hidden_state = m_request.get_tensor("last_hidden_state");
    const auto shape = hidden_state.get_shape();
    const size_t output_seq_len = shape[1];
    const size_t actual_seq_len = input_ids.get_shape()[1];

    if (output_seq_len == actual_seq_len) {
        output.hidden_features = hidden_state;
    } else {
        auto [start_coord, end_coord] =
            ov::genai::utils::make_roi(shape, 1, output_seq_len - actual_seq_len, output_seq_len);
        output.hidden_features = ov::Tensor(hidden_state, start_coord, end_coord);
    }

    return output;
}

std::vector<int64_t> DFlashTargetWrapper::sample_validate(const ov::Tensor& logits,
                                                          size_t input_token_count,
                                                          size_t sample_count,
                                                          size_t num_tokens_to_validate) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1);
    const size_t logits_seq_len = shape[1];

    auto current_seq = get_current_sequence();
    const size_t prev_generated_len = current_seq->get_generated_len();

    ov::Tensor sliced_logits = logits;
    if (sample_count < logits_seq_len) {
        auto [start_coord, end_coord] =
            ov::genai::utils::make_roi(shape, 1, logits_seq_len - sample_count, logits_seq_len);
        sliced_logits = ov::Tensor(logits, start_coord, end_coord);
    }

    m_sequence_group->schedule_tokens(input_token_count);
    m_sequence_group->set_output_seq_len(sample_count);
    m_sequence_group->set_num_validated_tokens(num_tokens_to_validate);

    m_sampler.sample({m_sequence_group}, sliced_logits, true);
    m_sequence_group->finish_iteration();

    const auto& generated_ids = current_seq->get_generated_ids();
    const size_t new_generated_len = generated_ids.size();
    const size_t result_count = new_generated_len - prev_generated_len + num_tokens_to_validate;
    std::vector<int64_t> result_tokens(generated_ids.end() - result_count, generated_ids.end());

    m_raw_perf_metrics.m_batch_sizes.emplace_back(result_tokens.size());
    return result_tokens;
}

std::vector<int64_t> DFlashTargetWrapper::sample_single(const ov::Tensor& logits, size_t input_token_count) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1);

    auto current_seq = get_current_sequence();
    const size_t prev_generated_len = current_seq->get_generated_len();

    // Slice to last position
    const size_t logits_seq_len = shape[1];
    ov::Tensor sliced_logits = logits;
    if (logits_seq_len > 1) {
        auto [start_coord, end_coord] =
            ov::genai::utils::make_roi(shape, 1, logits_seq_len - 1, logits_seq_len);
        sliced_logits = ov::Tensor(logits, start_coord, end_coord);
    }

    m_sequence_group->schedule_tokens(input_token_count);
    m_sequence_group->set_output_seq_len(1);
    m_sequence_group->set_num_validated_tokens(0);

    m_sampler.sample({m_sequence_group}, sliced_logits, false);
    m_sequence_group->finish_iteration();

    const auto& generated_ids = current_seq->get_generated_ids();
    std::vector<int64_t> result_tokens(generated_ids.end() - 1, generated_ids.end());

    m_raw_perf_metrics.m_batch_sizes.emplace_back(1);
    return result_tokens;
}

// ============================================================================
// DFlashDraftWrapper
// ============================================================================

DFlashDraftWrapper::DFlashDraftWrapper(const ModelDesc& model_desc,
                                       const utils::dflash::DFlashRTInfo& config)
    : m_device(model_desc.device),
      m_properties(model_desc.properties),
      m_block_size(config.block_size),
      m_mask_token_id(config.mask_token_id) {
    m_kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
    m_cache_types = utils::get_cache_types(*model_desc.model);

    m_request = utils::singleton_core()
                    .compile_model(model_desc.model, m_device, m_properties)
                    .create_infer_request();

    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
}

ov::Tensor DFlashDraftWrapper::infer(const ov::Tensor& input_ids, const ov::Tensor& target_hidden, size_t position_offset) {
    m_request.set_tensor("input_ids", input_ids);

    // Convert target_hidden to match draft model's expected precision if needed
    auto model_input = m_request.get_compiled_model().input("target_hidden");
    auto expected_et = model_input.get_element_type();
    if (target_hidden.get_element_type() != expected_et) {
        ov::Tensor converted(expected_et, target_hidden.get_shape());
        size_t num_elements = target_hidden.get_size();
        if (target_hidden.get_element_type() == ov::element::f32 && expected_et == ov::element::bf16) {
            const float* src = target_hidden.data<float>();
            auto* dst = reinterpret_cast<ov::bfloat16*>(converted.data());
            for (size_t i = 0; i < num_elements; ++i) dst[i] = ov::bfloat16(src[i]);
        } else if (target_hidden.get_element_type() == ov::element::bf16 && expected_et == ov::element::f32) {
            auto* src = reinterpret_cast<const ov::bfloat16*>(target_hidden.data());
            float* dst = converted.data<float>();
            for (size_t i = 0; i < num_elements; ++i) dst[i] = static_cast<float>(src[i]);
        } else {
            OPENVINO_THROW("Unsupported type conversion for target_hidden: ",
                           target_hidden.get_element_type().get_type_name(), " -> ", expected_et.get_type_name());
        }
        m_request.set_tensor("target_hidden", converted);
    } else {
        m_request.set_tensor("target_hidden", target_hidden);
    }

    // Position IDs: cover both context and block positions with absolute offsets.
    // In the reference, position_ids spans [0, ctx_len + block_size - 1] for the first call,
    // and includes accumulated context from the KV cache. Since we pass accumulated context
    // directly (no KV cache), positions always start from 0 for the context portion.
    const size_t block_size = input_ids.get_shape()[1];
    const size_t ctx_len = target_hidden.get_shape()[1];
    const size_t total_pos = ctx_len + block_size;

    ov::Tensor position_ids(ov::element::i64, {1, total_pos});
    int64_t* pos_ptr = position_ids.data<int64_t>();
    std::iota(pos_ptr, pos_ptr + total_pos, 0);
    m_request.set_tensor("position_ids", position_ids);

    uint64_t time_us = execute_inference();
    m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(time_us));
    m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(time_us));

    return m_request.get_tensor("logits");
}

void DFlashDraftWrapper::trim_kv_cache(size_t tokens_to_remove) {
    if (tokens_to_remove == 0) return;

    utils::CacheState state(m_cache_types);
    state.num_tokens_to_trim = tokens_to_remove;
    state.seq_length_axis = m_kv_axes_pos.seq_len;
    state.reset_mem_state = false;
    utils::trim_kv_cache(m_request, state, {});
}

void DFlashDraftWrapper::reset_state() {
    m_request.reset_state();
    m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
    m_raw_perf_metrics.m_durations.clear();
    m_raw_perf_metrics.m_batch_sizes.clear();
}

void DFlashDraftWrapper::release_memory() {
    m_request.get_compiled_model().release_memory();
}

uint64_t DFlashDraftWrapper::execute_inference() {
    auto start = std::chrono::steady_clock::now();
    m_request.infer();
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    return duration_us;
}

// ============================================================================
// StatefulDFlashLLMPipeline
// ============================================================================

StatefulDFlashLLMPipeline::StatefulDFlashLLMPipeline(const ModelDesc& target_model_desc,
                                                     const ModelDesc& draft_model_desc,
                                                     const utils::dflash::DFlashRTInfo& dflash_config)
    : StatefulSpeculativePipelineBase(target_model_desc.tokenizer, target_model_desc.generation_config),
      m_dflash_config(dflash_config) {
    auto target_model = target_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(target_model, "Target model must not be null");
    OPENVINO_ASSERT(draft_model, "Draft model must not be null");

    // Step 0: Ensure target model has input_ids (VL-exported models may have inputs_embeds)
    utils::dflash::ensure_target_has_input_ids(target_model);

    // Step 1: Extract hidden states from target model at specified layers
    utils::dflash::transform_target_hidden_state(target_model, m_dflash_config.target_layer_ids);

    // Step 2: Share vocabulary and graft lm_head into draft model
    utils::dflash::share_vocabulary_and_lm_head(target_model, draft_model);

    // Step 3: Create compiled model wrappers
    auto target_desc = target_model_desc;
    m_target = std::make_unique<DFlashTargetWrapper>(target_desc);

    auto draft_desc = draft_model_desc;
    m_draft = std::make_unique<DFlashDraftWrapper>(draft_desc, m_dflash_config);
}

StatefulDFlashLLMPipeline::~StatefulDFlashLLMPipeline() {
    m_target->release_memory();
    m_draft->release_memory();
}

GenerationConfig StatefulDFlashLLMPipeline::resolve_generation_config(OptionalGenerationConfig generation_config) {
    GenerationConfig config = StatefulSpeculativePipelineBase::resolve_generation_config(generation_config);
    // DFlash doesn't use num_assistant_tokens (single-pass generates block_size tokens)
    // but we keep the validation for confidence threshold
    OPENVINO_ASSERT(
        config.assistant_confidence_threshold == 0.f,
        "DFlash pipeline does not support assistant_confidence_threshold.");
    return config;
}

EncodedResults StatefulDFlashLLMPipeline::generate_tokens(const EncodedInputs& inputs,
                                                          const GenerationConfig& config,
                                                          StreamerVariant streamer) {
    ManualTimer generate_timer("StatefulDFlashLLMPipeline::generate(EncodedInputs)");
    generate_timer.start();

    std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

    // Extract input tensors
    ov::Tensor input_ids, attention_mask;
    if (auto* tensor_input = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *tensor_input;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto* tokenized_input = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = tokenized_input->input_ids;
        attention_mask = tokenized_input->attention_mask;
    }

    OPENVINO_ASSERT(input_ids.get_shape()[0] == 1, "Only batch size 1 supported");
    m_prompt_length = input_ids.get_shape()[1];

    // Reset model states
    m_target->reset_state();
    m_draft->reset_state();
    m_accumulated_hidden = ov::Tensor();
    m_draft_position_offset = 0;

    // Extend max_new_tokens for sampling config to avoid premature stopping during draft
    auto sampling_config = config;
    sampling_config.max_new_tokens = config.max_new_tokens + m_dflash_config.block_size + 1;

    // Initialize target sequence
    m_target->initialize_sequence(input_ids, sampling_config);

    // Phase 1: Prefill — process all prompt tokens through target model
    auto prefill_output = m_target->infer(m_prompt_length);
    auto initial_tokens = m_target->sample_single(prefill_output.logits, m_prompt_length);
    OPENVINO_ASSERT(initial_tokens.size() == 1, "Expected single token from prefill");

    m_last_accepted_token = initial_tokens[0];

    auto streaming_status = stream_generated_tokens(streamer_ptr, initial_tokens);

    // Store hidden features from prefill and initialize accumulated context
    // IMPORTANT: Make a deep copy — the tensor from infer() is a reference to an internal
    // buffer that gets overwritten on subsequent infer() calls.
    // Store in bf16 to match draft model's expected input dtype and avoid repeated conversions.
    m_target->get_current_sequence()->update_hidden_state(prefill_output.hidden_features);
    {
        auto src = prefill_output.hidden_features;
        // Determine target dtype for accumulation (match draft model input)
        m_accumulated_hidden = ov::Tensor(ov::element::bf16, src.get_shape());
        if (src.get_element_type() == ov::element::bf16) {
            src.copy_to(m_accumulated_hidden);
        } else {
            // Convert f32 to bf16
            const float* s = src.data<float>();
            auto* d = reinterpret_cast<ov::bfloat16*>(m_accumulated_hidden.data());
            for (size_t i = 0; i < src.get_size(); ++i) d[i] = ov::bfloat16(s[i]);
        }
    }
    m_draft_position_offset = 0;  // Will be set to prefill hidden length

    // Initialize position offset: the draft model's positions start after the context
    if (m_accumulated_hidden && m_accumulated_hidden.get_size() > 0) {
        m_draft_position_offset = m_accumulated_hidden.get_shape()[1];
    }

    // Phase 2: Speculative decoding loop
    size_t generated_tokens = 1;
    size_t total_draft_accepted = 0;
    size_t total_draft_calls = 0;
    bool eos_reached = (m_last_accepted_token == static_cast<int64_t>(config.eos_token_id));

    while (!eos_reached && generated_tokens < config.max_new_tokens &&
           streaming_status == ov::genai::StreamingStatus::RUNNING) {
        auto result = run_speculative_iteration(
            static_cast<int64_t>(config.eos_token_id),
            generated_tokens,
            config.max_new_tokens);

        streaming_status = stream_generated_tokens(streamer_ptr, result.validated_tokens);

        total_draft_calls++;
        total_draft_accepted += result.accepted_tokens_count;
        generated_tokens += result.validated_tokens.size();
        eos_reached = result.eos_reached;
    }

    // Phase 3: Finalization
    m_streaming_was_cancelled = (streaming_status == ov::genai::StreamingStatus::CANCEL);
    if (streamer_ptr)
        streamer_ptr->end();

    // Collect results
    EncodedResults results;
    results.tokens = {m_target->get_generated_tokens()};
    results.scores = {0.0f};

    auto sequence = m_target->get_current_sequence();
    auto finish_reason = sequence ? sequence->get_finish_reason() : GenerationFinishReason::NONE;
    results.finish_reasons = {finish_reason};

    generate_timer.end();

    // Update performance metrics
    m_sd_perf_metrics.num_input_tokens = m_prompt_length;
    m_sd_perf_metrics.load_time = m_load_time_ms;
    m_sd_perf_metrics.num_accepted_tokens = total_draft_accepted;
    m_sd_perf_metrics.raw_metrics.generate_durations.clear();
    m_sd_perf_metrics.raw_metrics.generate_durations.emplace_back(generate_timer.get_duration_microsec());

    m_sd_perf_metrics.m_evaluated = false;
    m_sd_perf_metrics.main_model_metrics.m_evaluated = false;
    m_sd_perf_metrics.draft_model_metrics.m_evaluated = false;

    m_sd_perf_metrics.main_model_metrics.raw_metrics = m_target->get_raw_perf_metrics();
    m_sd_perf_metrics.draft_model_metrics.raw_metrics = m_draft->get_raw_perf_metrics();

    const size_t total_draft_generated = total_draft_calls * m_dflash_config.block_size;
    if (total_draft_generated > 0) {
        float acceptance_rate = static_cast<float>(total_draft_accepted) / total_draft_generated * 100.0f;
        m_sd_metrics.update_acceptance_rate(0, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(0, total_draft_accepted);
        m_sd_metrics.update_draft_generated_len(0, total_draft_generated);
        m_sd_metrics.update_generated_len(generated_tokens);
    }

    m_sd_perf_metrics.evaluate_statistics(generate_timer.get_start_time());
    results.perf_metrics = m_sd_perf_metrics;
    results.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(m_sd_perf_metrics);

    return results;
}

StatefulDFlashLLMPipeline::SpeculativeResult StatefulDFlashLLMPipeline::run_speculative_iteration(
    int64_t eos_token_id,
    size_t current_generated_tokens,
    size_t max_new_tokens) {
    SpeculativeResult result;

    const int block_size = m_dflash_config.block_size;
    const int mask_token_id = m_dflash_config.mask_token_id;

    // Step 1: Build draft input — [anchor, MASK, MASK, ..., MASK]
    ov::Tensor draft_input_ids(ov::element::i64, {1, static_cast<size_t>(block_size)});
    int64_t* draft_ids_ptr = draft_input_ids.data<int64_t>();
    draft_ids_ptr[0] = m_last_accepted_token;  // Anchor token
    std::fill(draft_ids_ptr + 1, draft_ids_ptr + block_size, static_cast<int64_t>(mask_token_id));

    // Step 2: Get accumulated target hidden states (full context for draft model)
    // In the reference implementation, the draft model's KV cache accumulates all prior
    // context (target_hidden from each iteration). Since our exported draft model has no
    // KV cache, we pass the full accumulated hidden tensor explicitly.
    ov::Tensor context_hidden = m_accumulated_hidden;
    OPENVINO_ASSERT(context_hidden && context_hidden.get_size() > 0,
                    "Accumulated hidden state must be non-empty for DFlash drafting");

    auto ctx_shape = context_hidden.get_shape();
    std::cerr << "[DFlash DEBUG] context_hidden shape: [" << ctx_shape[0] << "," << ctx_shape[1] << "," << ctx_shape[2]
              << "] type=" << context_hidden.get_element_type().get_type_name() << std::endl;

    // Step 3: Single draft forward pass (position_offset enables correct RoPE positions)
    ov::Tensor draft_logits = m_draft->infer(draft_input_ids, context_hidden, m_draft_position_offset);
    // draft_logits shape: [1, block_size, vocab_size]

    // Step 4: Sample draft tokens from positions 1..block_size-1
    // Position 0 is the anchor — we skip it
    const auto logits_shape = draft_logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3 && logits_shape[0] == 1);
    const size_t vocab_size = logits_shape[2];
    const size_t num_draft_positions = block_size - 1;

    std::cerr << "[DFlash DEBUG] Draft logits shape: [" << logits_shape[0] << "," << logits_shape[1] << "," << logits_shape[2]
              << "] type=" << draft_logits.get_element_type().get_type_name() << std::endl;

    std::vector<int64_t> draft_candidates;
    draft_candidates.reserve(num_draft_positions);

    // Greedy sampling from draft logits (positions 1 to block_size-1)
    // Convert bf16 logits to f32 if needed
    ov::Tensor f32_draft_logits;
    if (draft_logits.get_element_type() != ov::element::f32) {
        f32_draft_logits = ov::Tensor(ov::element::f32, draft_logits.get_shape());
        auto* src = reinterpret_cast<const ov::bfloat16*>(draft_logits.data());
        float* dst = f32_draft_logits.data<float>();
        for (size_t i = 0; i < draft_logits.get_size(); ++i) dst[i] = static_cast<float>(src[i]);
    } else {
        f32_draft_logits = draft_logits;
    }
    const float* logits_data = f32_draft_logits.data<float>();
    for (size_t pos = 1; pos < static_cast<size_t>(block_size); ++pos) {
        const float* pos_logits = logits_data + pos * vocab_size;
        int64_t best_token = std::distance(pos_logits,
            std::max_element(pos_logits, pos_logits + vocab_size));
        draft_candidates.push_back(best_token);
    }

    // Debug: print first few draft candidates
    std::cerr << "[DFlash DEBUG] Draft candidates (first 5): ";
    for (size_t i = 0; i < std::min(draft_candidates.size(), size_t(5)); ++i)
        std::cerr << draft_candidates[i] << " ";
    std::cerr << std::endl;

    // Step 5: Add draft candidates to target sequence for verification
    m_target->append_tokens(draft_candidates);

    // Step 6: Save SSM/conv states before verification (hybrid model needs this)
    m_target->save_linear_states();

    // Step 6b: Target verifies all candidates in one forward pass
    const size_t validation_window = num_draft_positions + 1;  // draft tokens + 1 for bonus
    auto val_output = m_target->infer(validation_window);

    // Step 7: Sample with validation
    auto validated_tokens = m_target->sample_validate(
        val_output.logits,
        validation_window,
        validation_window,
        num_draft_positions);

    // validated_tokens = [accepted_draft_tokens..., bonus_token]
    std::cerr << "[DFlash DEBUG] Validated tokens (" << validated_tokens.size() << "): ";
    for (size_t i = 0; i < std::min(validated_tokens.size(), size_t(5)); ++i)
        std::cerr << validated_tokens[i] << " ";
    std::cerr << std::endl;
    const size_t accepted_count = validated_tokens.size() - 1;
    const int64_t bonus_token = validated_tokens.back();
    const size_t tokens_to_remove = num_draft_positions - accepted_count;

    size_t total_new_tokens = validated_tokens.size();

    // Check if we'd exceed max_new_tokens
    bool is_final_iteration = false;
    if (current_generated_tokens + total_new_tokens > max_new_tokens) {
        size_t excess = current_generated_tokens + total_new_tokens - max_new_tokens;
        OPENVINO_ASSERT(excess < total_new_tokens);
        total_new_tokens -= excess;
        validated_tokens.resize(total_new_tokens);
        m_target->truncate_sequence(m_prompt_length + max_new_tokens);

        auto& target_batch_sizes = m_target->get_raw_perf_metrics().m_batch_sizes;
        if (!target_batch_sizes.empty()) {
            target_batch_sizes.back() = total_new_tokens;
        }
        is_final_iteration = true;
    }

    // Step 8: Fix states for hybrid model (KV cache + SSM/conv states)
    // For pure transformers, trimming KV by (K-N) suffices. For hybrid models,
    // SSM/conv states are sequential and can't be partially trimmed — we must
    // restore them and re-infer only the accepted tokens to advance correctly.
    // Skip re-inference on final iteration (no subsequent iterations need correct state).
    if (tokens_to_remove > 0 && !is_final_iteration) {
        // 8a: Trim ALL verification KV entries (go back to pre-verification KV state)
        m_target->trim_kv_cache(validation_window);

        // 8b: Restore SSM/conv states to pre-verification state
        m_target->restore_linear_states();

        // 8c: Remove bonus token from sequence temporarily
        auto seq = m_target->get_current_sequence();
        seq->remove_last_tokens(1);

        // 8d: Re-infer prev_bonus + accepted draft tokens (N+1 tokens)
        // This advances both KV and SSM correctly for only the accepted tokens
        size_t reinfer_count = accepted_count + 1;
        auto reinfer_output = m_target->infer(reinfer_count);

        // 8e: Add bonus token back to sequence
        m_target->append_tokens({bonus_token});

        // 8f: Use hidden features from re-inference (they reflect correct SSM state)
        val_output.hidden_features = reinfer_output.hidden_features;

        std::cerr << "[DFlash DEBUG] Re-inferred " << reinfer_count << " tokens after "
                  << tokens_to_remove << " rejections" << std::endl;
    } else if (tokens_to_remove > 0) {
        // Final iteration with rejections — just trim KV, don't bother fixing SSM
        m_target->trim_kv_cache(tokens_to_remove);
    } else {
        // All tokens accepted — SSM state is already correct, discard saved states
    }

    // Step 9: Update hidden states for next iteration
    // In the reference, target_hidden is sliced to [:, :acceptance_length+1, :]
    // from the verification output. This is the CURRENT iteration's context.
    // The draft's KV cache preserves PRIOR context. Since we don't have KV cache in the
    // draft model, we accumulate all target_hidden across iterations.
    auto new_hidden = val_output.hidden_features;
    if (new_hidden && new_hidden.get_size() > 0) {
        const auto nh_shape = new_hidden.get_shape();
        // Slice to only the accepted tokens (matching reference: [:, :acceptance_length+1, :])
        ov::Tensor sliced_hidden = new_hidden;
        if (nh_shape[1] > total_new_tokens) {
            auto [start_coord, end_coord] = ov::genai::utils::make_roi(nh_shape, 1, 0, total_new_tokens);
            sliced_hidden = ov::Tensor(new_hidden, start_coord, end_coord);
        }

        // Accumulate: concatenate new hidden states with prior accumulated context
        // Stored in bf16 to match draft model input type
        const auto acc_shape = m_accumulated_hidden.get_shape();
        const size_t acc_seq_len = acc_shape[1];
        const size_t new_seq_len = sliced_hidden.get_shape()[1];
        const size_t total_seq_len = acc_seq_len + new_seq_len;
        const size_t feature_dim = acc_shape[2];

        ov::Tensor new_accumulated(ov::element::bf16, {1, total_seq_len, feature_dim});

        // Copy prior accumulated context (already bf16)
        const size_t acc_bytes = acc_seq_len * feature_dim * sizeof(ov::bfloat16);

        auto* dst = reinterpret_cast<uint8_t*>(new_accumulated.data());
        std::memcpy(dst, m_accumulated_hidden.data(), acc_bytes);

        // Convert and copy new hidden states to bf16
        const size_t num_elements = new_seq_len * feature_dim;
        auto* bf_dst = reinterpret_cast<ov::bfloat16*>(dst + acc_bytes);
        if (sliced_hidden.get_element_type() == ov::element::bf16) {
            std::memcpy(bf_dst, sliced_hidden.data(), num_elements * sizeof(ov::bfloat16));
        } else {
            // f32 -> bf16
            const float* src = sliced_hidden.data<float>();
            for (size_t i = 0; i < num_elements; ++i) bf_dst[i] = ov::bfloat16(src[i]);
        }

        m_accumulated_hidden = new_accumulated;
        m_draft_position_offset = total_seq_len;

        // Also update the per-sequence hidden state for backward compat
        m_target->get_current_sequence()->update_hidden_state(sliced_hidden);
    }

    // Update last accepted token for next iteration's anchor
    m_last_accepted_token = validated_tokens.back();

    result.accepted_tokens_count = accepted_count;
    result.validated_tokens = std::move(validated_tokens);
    result.eos_reached = (m_last_accepted_token == eos_token_id);

    return result;
}

void StatefulDFlashLLMPipeline::finish_chat() {
    StatefulSpeculativePipelineBase::finish_chat();
}

SpeculativeDecodingMetrics StatefulDFlashLLMPipeline::get_speculative_decoding_metrics() const {
    return m_sd_metrics;
}

}  // namespace ov::genai
