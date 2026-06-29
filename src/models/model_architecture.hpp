#pragma once
// =============================================================================
// model_architecture.hpp — LLM Structural Profiles and Quantization
// =============================================================================
//
// Defines the exact architectural dimensions needed to calculate model weights
// and KV cache memory constraints correctly for DES inference simulation.
// =============================================================================

#include <string>

namespace llmsimbench {

/// Data types for weight and KV cache quantization
enum class DataType {
    FP16, // 16-bit float (2 bytes)
    BF16, // 16-bit bfloat (2 bytes)
    FP8,  // 8-bit float (1 byte)
    INT8, // 8-bit integer (1 byte)
    INT4, // 4-bit integer (0.5 bytes)
};

/// Get the number of bytes per parameter for a given data type
inline double get_bytes_per_param(DataType dtype) {
    switch (dtype) {
        case DataType::FP16: return 2.0;
        case DataType::BF16: return 2.0;
        case DataType::FP8:  return 1.0;
        case DataType::INT8: return 1.0;
        case DataType::INT4: return 0.5;
        default: return 2.0;
    }
}

/// Structural dimensions of a Transformer-based LLM
struct ModelArchitecture {
    std::string name;
    double params_B;       // Total parameter count in billions
    int num_layers;        // Number of transformer layers
    int num_kv_heads;      // Number of Key/Value attention heads (GQA support)
    int head_dimension;    // Dimension of each attention head
    int hidden_size;       // Hidden dimension size
    int vocab_size;        // Vocabulary size
};

// ── Predefined standard models ───────────────────────────────────────────────

inline ModelArchitecture LLAMA_3_8B_PROFILE() {
    return {
        .name           = "Llama-3-8B",
        .params_B       = 8.03,
        .num_layers     = 32,
        .num_kv_heads   = 8,      // GQA (32 Query, 8 KV)
        .head_dimension = 128,
        .hidden_size    = 4096,
        .vocab_size     = 128256
    };
}

inline ModelArchitecture LLAMA_3_70B_PROFILE() {
    return {
        .name           = "Llama-3-70B",
        .params_B       = 70.6,
        .num_layers     = 80,
        .num_kv_heads   = 8,      // GQA (64 Query, 8 KV)
        .head_dimension = 128,
        .hidden_size    = 8192,
        .vocab_size     = 128256
    };
}

inline ModelArchitecture MISTRAL_7B_V01_PROFILE() {
    return {
        .name           = "Mistral-7B-v0.1",
        .params_B       = 7.24,
        .num_layers     = 32,
        .num_kv_heads   = 8,      // GQA (32 Query, 8 KV)
        .head_dimension = 128,
        .hidden_size    = 4096,
        .vocab_size     = 32000
    };
}

inline ModelArchitecture MIXTRAL_8X7B_PROFILE() {
    return {
        .name           = "Mixtral-8x7B",
        .params_B       = 46.7,   // Sparse parameter count
        .num_layers     = 32,
        .num_kv_heads   = 8,
        .head_dimension = 128,
        .hidden_size    = 4096,
        .vocab_size     = 32000
    };
}

inline ModelArchitecture QWEN_15_72B_PROFILE() {
    return {
        .name           = "Qwen1.5-72B",
        .params_B       = 72.7,
        .num_layers     = 80,
        .num_kv_heads   = 8,      // GQA (64 Query, 8 KV)
        .head_dimension = 128,
        .hidden_size    = 8192,
        .vocab_size     = 152064
    };
}

inline ModelArchitecture QWEN_2_7B_PROFILE() {
    return {
        .name           = "Qwen2-7B",
        .params_B       = 7.62,
        .num_layers     = 28,
        .num_kv_heads   = 4,      // GQA (28 Query, 4 KV)
        .head_dimension = 128,
        .hidden_size    = 3584,
        .vocab_size     = 152064
    };
}

inline ModelArchitecture PHI_3_MINI_PROFILE() {
    return {
        .name           = "Phi-3-Mini",
        .params_B       = 3.82,
        .num_layers     = 32,
        .num_kv_heads   = 32,     // MHA
        .head_dimension = 96,
        .hidden_size    = 3072,
        .vocab_size     = 32064
    };
}

/// Construct a custom architecture from Python.
inline ModelArchitecture make_architecture(
    const std::string& name, double params_b, int num_layers,
    int num_kv_heads, int head_dimension, int hidden_size, int vocab_size)
{
    return {name, params_b, num_layers, num_kv_heads, head_dimension, hidden_size, vocab_size};
}

} // namespace llmsimbench
