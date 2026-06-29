// =============================================================================
// pybind_module.cpp — pybind11 bindings for LLM Sim Bench core
// =============================================================================
//
// Exposes the entire C++ simulation core to Python as the `llmsimbench_core`
// module.  Key design decisions:
//
//   • std::shared_ptr holders for cross-boundary lifetime safety.
//   • GIL release on SimulationEngine::step() and run() for threading.
//   • Input validation on Python-facing constructors (ValueError on bad args).
//   • Zero-copy stats access via direct struct binding.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // automatic std::vector / std::map conversion
#include <pybind11/functional.h>   // std::function support

#include "../core/event.hpp"
#include "../core/event_queue.hpp"
#include "../core/workload.hpp"
#include "../core/scheduler.hpp"
#include "../models/hardware_profile.hpp"
#include "../models/model_architecture.hpp"
#include "../models/timing_model.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// ── Input validation helpers ────────────────────────────────────────────────

static void validate_positive(int value, const char* name) {
    if (value <= 0) {
        throw std::invalid_argument(
            std::string(name) + " must be a positive integer, got " +
            std::to_string(value));
    }
}

static void validate_non_negative(double value, const char* name) {
    if (value < 0.0) {
        throw std::invalid_argument(
            std::string(name) + " must be non-negative, got " +
            std::to_string(value));
    }
}

// =============================================================================
//  Module definition
// =============================================================================

PYBIND11_MODULE(llmsimbench_core, m) {
    m.doc() = "LLM Sim Bench — High-fidelity behavioral simulator for LLM "
              "inference serving systems.  C++20 core with pybind11 bindings.";

    // ─── EventType enum ─────────────────────────────────────────────────
    py::enum_<llmsimbench::EventType>(m, "EventType")
        .value("REQUEST_ARRIVAL",      llmsimbench::EventType::REQUEST_ARRIVAL)
        .value("PREFILL_COMPLETE",     llmsimbench::EventType::PREFILL_COMPLETE)
        .value("PREFILL_CHUNK_COMPLETE", llmsimbench::EventType::PREFILL_CHUNK_COMPLETE)
        .value("KV_TRANSFER_COMPLETE", llmsimbench::EventType::KV_TRANSFER_COMPLETE)
        .value("DECODE_STEP_COMPLETE", llmsimbench::EventType::DECODE_STEP_COMPLETE)
        .value("DECODE_COMPLETE",      llmsimbench::EventType::DECODE_COMPLETE)
        .value("PREEMPTION",          llmsimbench::EventType::PREEMPTION)
        .value("CACHE_EVICTION",      llmsimbench::EventType::CACHE_EVICTION)
        .export_values();

    // ─── RequestState enum ──────────────────────────────────────────────
    py::enum_<llmsimbench::RequestState>(m, "RequestState")
        .value("PENDING",    llmsimbench::RequestState::PENDING)
        .value("IN_PREFILL", llmsimbench::RequestState::IN_PREFILL)
        .value("IN_DECODE",  llmsimbench::RequestState::IN_DECODE)
        .value("PREEMPTED",  llmsimbench::RequestState::PREEMPTED)
        .value("COMPLETED",  llmsimbench::RequestState::COMPLETED)
        .export_values();

    // ─── Request ────────────────────────────────────────────────────────
    py::class_<llmsimbench::Request, std::shared_ptr<llmsimbench::Request>>(m, "Request")
        .def(py::init([](int id, int prompt_len, int gen_len, double arrival) {
            validate_positive(prompt_len, "prompt_length");
            validate_positive(gen_len,    "generation_length");
            validate_non_negative(arrival, "arrival_time");
            return std::make_shared<llmsimbench::Request>(
                id, prompt_len, gen_len, arrival);
        }),
            py::arg("id"),
            py::arg("prompt_length"),
            py::arg("generation_length"),
            py::arg("arrival_time") = 0.0,
            "Create a new inference request.\n\n"
            "Args:\n"
            "    id: Unique request identifier.\n"
            "    prompt_length: Number of tokens in the input prompt.\n"
            "    generation_length: Number of tokens to generate.\n"
            "    arrival_time: Simulation time when the request arrives (seconds).")
        .def("get_id",               &llmsimbench::Request::get_id)
        .def("get_prompt_length",    &llmsimbench::Request::get_prompt_length)
        .def("get_generation_length",&llmsimbench::Request::get_generation_length)
        .def("get_tokens_generated", &llmsimbench::Request::get_tokens_generated)
        .def("get_arrival_time",     &llmsimbench::Request::get_arrival_time)
        .def("get_state",           &llmsimbench::Request::get_state)
        .def("get_state_str",       &llmsimbench::Request::get_state_str)
        .def("is_completed",        &llmsimbench::Request::is_completed)
        .def("get_prefill_latency", &llmsimbench::Request::get_prefill_latency)
        .def("get_decode_latency",  &llmsimbench::Request::get_decode_latency)
        .def("get_end_to_end_latency", &llmsimbench::Request::get_end_to_end_latency)
        .def("get_prefill_start",   &llmsimbench::Request::get_prefill_start)
        .def("get_decode_start",    &llmsimbench::Request::get_decode_start)
        .def("get_completion_time", &llmsimbench::Request::get_completion_time)
        .def("get_preemption_count", &llmsimbench::Request::get_preemption_count)
        .def("get_prefill_tokens_processed", &llmsimbench::Request::get_prefill_tokens_processed)
        .def("get_remaining_prefill_tokens", &llmsimbench::Request::get_remaining_prefill_tokens)
        .def("__repr__", [](const llmsimbench::Request& r) {
            return "<Request id=" + std::to_string(r.get_id()) +
                   " state=" + r.get_state_str() +
                   " prompt=" + std::to_string(r.get_prompt_length()) +
                   " gen=" + std::to_string(r.get_generation_length()) + ">";
        });

    // ─── KVCacheEntry ───────────────────────────────────────────────────
    py::class_<llmsimbench::KVCacheEntry>(m, "KVCacheEntry")
        .def_readwrite("token_index",     &llmsimbench::KVCacheEntry::token_index)
        .def_readwrite("attention_score", &llmsimbench::KVCacheEntry::attention_score)
        .def_readwrite("last_access_time",&llmsimbench::KVCacheEntry::last_access_time)
        .def_readwrite("memory_bytes",    &llmsimbench::KVCacheEntry::memory_bytes);

    // ─── KVCache ────────────────────────────────────────────────────────
    py::class_<llmsimbench::KVCache, std::shared_ptr<llmsimbench::KVCache>>(m, "KVCache")
        .def(py::init<size_t, size_t>(),
             py::arg("capacity") = 4096,
             py::arg("bytes_per_entry") = 2048)
        .def("current_size",   &llmsimbench::KVCache::current_size)
        .def("capacity",       &llmsimbench::KVCache::capacity)
        .def("cache_hits",     &llmsimbench::KVCache::cache_hits)
        .def("cache_misses",   &llmsimbench::KVCache::cache_misses)
        .def("memory_bytes",   &llmsimbench::KVCache::memory_bytes)
        .def("hit_rate",       &llmsimbench::KVCache::hit_rate)
        .def("eviction_policy_name", &llmsimbench::KVCache::eviction_policy_name)
        .def("generate_kv_cache",    &llmsimbench::KVCache::generate_kv_cache,
             py::arg("num_tokens"), py::arg("sim_time"))
        .def("access_token",        &llmsimbench::KVCache::access_token,
             py::arg("token_idx"), py::arg("sim_time"))
        .def("add_decode_token",    &llmsimbench::KVCache::add_decode_token,
             py::arg("token_idx"), py::arg("sim_time"))
        .def("clear",               &llmsimbench::KVCache::clear);

    // ─── Eviction Policies ──────────────────────────────────────────────
    py::class_<llmsimbench::EvictionPolicy,
               std::shared_ptr<llmsimbench::EvictionPolicy>>(m, "EvictionPolicy")
        .def("name", &llmsimbench::EvictionPolicy::name);

    py::class_<llmsimbench::RandomEvictionPolicy,
               llmsimbench::EvictionPolicy,
               std::shared_ptr<llmsimbench::RandomEvictionPolicy>>(m, "RandomEvictionPolicy")
        .def(py::init<>());

    py::class_<llmsimbench::LRUEvictionPolicy,
               llmsimbench::EvictionPolicy,
               std::shared_ptr<llmsimbench::LRUEvictionPolicy>>(m, "LRUEvictionPolicy")
        .def(py::init<>());

    py::class_<llmsimbench::AttentionGuidedEvictionPolicy,
               llmsimbench::EvictionPolicy,
               std::shared_ptr<llmsimbench::AttentionGuidedEvictionPolicy>>(
                   m, "AttentionGuidedEvictionPolicy")
        .def(py::init<>());

    // ─── HardwareProfile ────────────────────────────────────────────────
    py::class_<llmsimbench::HardwareProfile>(m, "HardwareProfile")
        .def(py::init<>())
        .def_readwrite("name",                   &llmsimbench::HardwareProfile::name)
        .def_readwrite("compute_tflops",         &llmsimbench::HardwareProfile::compute_tflops)
        .def_readwrite("memory_bandwidth_gbps",  &llmsimbench::HardwareProfile::memory_bandwidth_gbps)
        .def_readwrite("memory_capacity_gb",     &llmsimbench::HardwareProfile::memory_capacity_gb)
        .def_readwrite("network_bandwidth_gbps", &llmsimbench::HardwareProfile::network_bandwidth_gbps)
        .def_readwrite("network_latency_ms",     &llmsimbench::HardwareProfile::network_latency_ms)
        .def("__repr__", [](const llmsimbench::HardwareProfile& h) {
            return "<HardwareProfile '" + h.name +
                   "' compute=" + std::to_string(h.compute_tflops) +
                   " TFLOPS  mem_bw=" + std::to_string(h.memory_bandwidth_gbps) +
                   " GB/s>";
        });

    // Factory functions for predefined profiles
    m.def("H100_PROFILE", &llmsimbench::H100_PROFILE,
          "NVIDIA H100 SXM 80GB hardware profile.");
    m.def("A100_80GB_PROFILE", &llmsimbench::A100_80GB_PROFILE, "Get standard A100 80GB profile");
    m.def("L4_PROFILE",        &llmsimbench::L4_PROFILE,        "Get standard L4 24GB profile");
    m.def("H200_PROFILE",      &llmsimbench::H200_PROFILE,      "Get standard H200 141GB profile");
    m.def("GB200_PROFILE",     &llmsimbench::GB200_PROFILE,     "Get standard GB200 192GB profile");
    m.def("GB300_PROFILE",     &llmsimbench::GB300_PROFILE,     "Get standard GB300 288GB profile");
    m.def("make_profile", &llmsimbench::make_profile,
          py::arg("name"), py::arg("compute_tflops"),
          py::arg("memory_bandwidth_gbps"), py::arg("memory_capacity_gb"),
          py::arg("network_bandwidth_gbps"), py::arg("network_latency_ms"),
          "Create a custom hardware profile.");

    // ─── DataType enum ───────────────────────────────────────────────────────
    py::enum_<llmsimbench::DataType>(m, "DataType")
        .value("FP16", llmsimbench::DataType::FP16)
        .value("BF16", llmsimbench::DataType::BF16)
        .value("FP8",  llmsimbench::DataType::FP8)
        .value("INT8", llmsimbench::DataType::INT8)
        .value("INT4", llmsimbench::DataType::INT4)
        .export_values();

    m.def("get_bytes_per_param", &llmsimbench::get_bytes_per_param, "Get bytes per parameter for DataType");

    // ─── ModelArchitecture ───────────────────────────────────────────────────
    py::class_<llmsimbench::ModelArchitecture>(m, "ModelArchitecture")
        .def(py::init<std::string, double, int, int, int, int, int>(),
             py::arg("name"), py::arg("params_B"), py::arg("num_layers"),
             py::arg("num_kv_heads"), py::arg("head_dimension"),
             py::arg("hidden_size"), py::arg("vocab_size"))
        .def_readwrite("name",           &llmsimbench::ModelArchitecture::name)
        .def_readwrite("params_B",       &llmsimbench::ModelArchitecture::params_B)
        .def_readwrite("num_layers",     &llmsimbench::ModelArchitecture::num_layers)
        .def_readwrite("num_kv_heads",   &llmsimbench::ModelArchitecture::num_kv_heads)
        .def_readwrite("head_dimension", &llmsimbench::ModelArchitecture::head_dimension)
        .def_readwrite("hidden_size",    &llmsimbench::ModelArchitecture::hidden_size)
        .def_readwrite("vocab_size",     &llmsimbench::ModelArchitecture::vocab_size)
        .def("__repr__", [](const llmsimbench::ModelArchitecture& ma) {
            return "<ModelArchitecture '" + ma.name + "' (" + std::to_string(ma.params_B) + "B)>";
        });

    m.def("LLAMA_3_8B_PROFILE",     &llmsimbench::LLAMA_3_8B_PROFILE);
    m.def("LLAMA_3_70B_PROFILE",    &llmsimbench::LLAMA_3_70B_PROFILE);
    m.def("MISTRAL_7B_V01_PROFILE", &llmsimbench::MISTRAL_7B_V01_PROFILE);
    m.def("MIXTRAL_8X7B_PROFILE",   &llmsimbench::MIXTRAL_8X7B_PROFILE);
    m.def("QWEN_15_72B_PROFILE",    &llmsimbench::QWEN_15_72B_PROFILE);
    m.def("QWEN_2_7B_PROFILE",      &llmsimbench::QWEN_2_7B_PROFILE);
    m.def("PHI_3_MINI_PROFILE",     &llmsimbench::PHI_3_MINI_PROFILE);

    // ─── Scheduler hierarchy ────────────────────────────────────────────
    py::class_<llmsimbench::Scheduler,
               std::shared_ptr<llmsimbench::Scheduler>>(m, "Scheduler")
        .def("policy_name", &llmsimbench::Scheduler::policy_name);

    py::class_<llmsimbench::FirstComeFirstServeScheduler,
               llmsimbench::Scheduler,
               std::shared_ptr<llmsimbench::FirstComeFirstServeScheduler>>(
                   m, "FirstComeFirstServeScheduler")
        .def(py::init<>());

    py::class_<llmsimbench::PriorityScheduler,
               llmsimbench::Scheduler,
               std::shared_ptr<llmsimbench::PriorityScheduler>>(
                   m, "PriorityScheduler")
        .def(py::init<double, double>(),
             py::arg("prompt_weight") = 0.1,
             py::arg("generation_weight") = 0.5);

    py::class_<llmsimbench::ContinuousBatchingScheduler,
               llmsimbench::Scheduler,
               std::shared_ptr<llmsimbench::ContinuousBatchingScheduler>>(
                   m, "ContinuousBatchingScheduler")
        .def(py::init<int>(), py::arg("max_batch_size") = 8);

    py::class_<llmsimbench::PriorityContinuousBatchingScheduler,
               llmsimbench::Scheduler,
               std::shared_ptr<llmsimbench::PriorityContinuousBatchingScheduler>>(
                   m, "PriorityContinuousBatchingScheduler")
        .def(py::init<int, double, double, double>(),
             py::arg("max_batch_size") = 8,
             py::arg("prompt_weight") = 0.1,
             py::arg("generation_weight") = 0.5,
             py::arg("preemption_threshold") = 0.5);

    // ─── SimulationStats ────────────────────────────────────────────────
    py::class_<llmsimbench::SimulationStats>(m, "SimulationStats")
        .def_readonly("total_requests_completed",  &llmsimbench::SimulationStats::total_requests_completed)
        .def_readonly("total_tokens_generated",    &llmsimbench::SimulationStats::total_tokens_generated)
        .def_readonly("total_prefill_time",        &llmsimbench::SimulationStats::total_prefill_time)
        .def_readonly("total_decode_time",         &llmsimbench::SimulationStats::total_decode_time)
        .def_readonly("total_e2e_latency",         &llmsimbench::SimulationStats::total_e2e_latency)
        .def_readonly("max_e2e_latency",           &llmsimbench::SimulationStats::max_e2e_latency)
        .def_readonly("total_cache_hits",          &llmsimbench::SimulationStats::total_cache_hits)
        .def_readonly("total_cache_misses",        &llmsimbench::SimulationStats::total_cache_misses)
        .def_readonly("peak_pending_queue_depth",  &llmsimbench::SimulationStats::peak_pending_queue_depth)
        .def_readonly("total_preemptions",          &llmsimbench::SimulationStats::total_preemptions)
        .def_readonly("total_memory_pressure_evictions", &llmsimbench::SimulationStats::total_memory_pressure_evictions)
        .def("avg_e2e_latency",          &llmsimbench::SimulationStats::avg_e2e_latency)
        .def("throughput_tokens_per_sec",&llmsimbench::SimulationStats::throughput_tokens_per_sec)
        .def("__repr__", [](const llmsimbench::SimulationStats& s) {
            return "<SimulationStats completed=" +
                   std::to_string(s.total_requests_completed) +
                   " tokens=" + std::to_string(s.total_tokens_generated) +
                   " avg_latency=" + std::to_string(s.avg_e2e_latency()) + "s>";
        });

    // ─── SimulationEngine ───────────────────────────────────────────────
    py::class_<llmsimbench::SimulationEngine,
               std::shared_ptr<llmsimbench::SimulationEngine>>(m, "SimulationEngine")
        .def(py::init([](std::shared_ptr<llmsimbench::Scheduler> scheduler,
                         llmsimbench::HardwareProfile prefill_hw,
                         llmsimbench::HardwareProfile decode_hw,
                         double model_params_B,
                         size_t kv_cache_capacity,
                         double kv_transfer_latency_ms,
                         int prefill_chunk_size,
                         size_t global_memory_budget_bytes) {
            if (!scheduler) {
                throw std::invalid_argument("scheduler must not be None");
            }
            validate_non_negative(model_params_B, "model_params_B");
            return std::make_shared<llmsimbench::SimulationEngine>(
                 std::move(scheduler),
                 std::move(prefill_hw), std::move(decode_hw),
                 model_params_B, kv_cache_capacity, kv_transfer_latency_ms,
                 prefill_chunk_size, global_memory_budget_bytes);
         }),
             py::arg("scheduler"),
             py::arg("prefill_hw"),
             py::arg("decode_hw"),
             py::arg("model_params_B") = 7.0,
             py::arg("kv_cache_capacity") = 4096,
             py::arg("kv_transfer_latency_ms") = 0.5,
             py::arg("prefill_chunk_size") = 0,
             py::arg("global_memory_budget_bytes") = 0,
            "Create a new simulation engine.\n\n"
            "Args:\n"
            "    scheduler: Scheduling policy (FCFS, Priority, etc.).\n"
            "    prefill_hw: Hardware profile for the prefill engine.\n"
            "    decode_hw: Hardware profile for the decode engine.\n"
            "    model_params_B: Model size in billions of parameters.\n"
            "    kv_cache_capacity: Max KV cache entries per request.\n"
            "    kv_transfer_latency_ms: Simulated PDD transfer latency (ms).")
        .def("add_request",
             &llmsimbench::SimulationEngine::add_request,
             py::arg("id"), py::arg("prompt_len"),
             py::arg("gen_len"), py::arg("arrival_time"),
             "Inject a new request into the simulation.")
        .def("add_request_obj",
             &llmsimbench::SimulationEngine::add_request_obj,
             py::arg("request"),
             "Inject a Request object into the simulation.")
        .def("step",
             [](llmsimbench::SimulationEngine& self) {
                 py::gil_scoped_release release;  // Release GIL during C++ work
                 return self.step();
             },
             "Process the next event. Returns False when simulation is complete.")
        .def("run",
             [](llmsimbench::SimulationEngine& self) {
                 py::gil_scoped_release release;
                 self.run();
             },
             "Run the simulation to completion.")
        .def("get_current_time",      &llmsimbench::SimulationEngine::get_current_time)
        .def("get_completed_requests",&llmsimbench::SimulationEngine::get_completed_requests)
        .def("get_all_requests",     &llmsimbench::SimulationEngine::get_all_requests)
        .def("pending_count",        &llmsimbench::SimulationEngine::pending_count)
        .def("completed_count",      &llmsimbench::SimulationEngine::completed_count)
        .def("total_request_count",  &llmsimbench::SimulationEngine::total_request_count)
        .def("stats",                &llmsimbench::SimulationEngine::stats,
             py::return_value_policy::reference_internal)
        .def("scheduler_name",      &llmsimbench::SimulationEngine::scheduler_name)
        .def("set_eviction_policy_for_all",
             &llmsimbench::SimulationEngine::set_eviction_policy_for_all,
             py::arg("policy_name"),
             "Set the KV cache eviction policy for all requests.\n"
             "Valid values: 'Random', 'LRU', 'AttentionGuided', or '' for unlimited.")
        .def("__repr__", [](const llmsimbench::SimulationEngine& e) {
            return "<SimulationEngine t=" +
                   std::to_string(e.get_current_time()) +
                   " requests=" + std::to_string(e.total_request_count()) +
                   " completed=" + std::to_string(e.completed_count()) + ">";
        });

    // ─── Module-level version info ──────────────────────────────────────
    m.attr("__version__") = "0.1.0";
    m.attr("__author__")  = "LLM Sim Bench Contributors";
}
