// =============================================================================
// test_des_engine.cpp — Unit tests for the LLM Sim Bench DES core
// =============================================================================
//
// Standalone test binary — no external test framework needed.
// Compile:  cmake --build build --target test_des_engine
// Run:      ./build/test_des_engine
// =============================================================================

#include "../src/core/event.hpp"
#include "../src/core/event_queue.hpp"
#include "../src/core/workload.hpp"
#include "../src/core/scheduler.hpp"
#include "../src/models/hardware_profile.hpp"
#include "../src/models/timing_model.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace llmsimbench;

// ── Test infrastructure ─────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

#define RUN_TEST(fn)                                        \
    do {                                                    \
        std::cout << "  " << #fn << " ... ";                \
        try {                                               \
            fn();                                           \
            std::cout << "PASS\n";                          \
            ++tests_passed;                                 \
        } catch (const std::exception& e) {                 \
            std::cout << "FAIL: " << e.what() << "\n";      \
            ++tests_failed;                                 \
        }                                                   \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) throw std::runtime_error(          \
        std::string("ASSERT_EQ failed: ") + #a + " (" +    \
        std::to_string(a) + ") != " + #b + " (" +          \
        std::to_string(b) + ")"); } while (0)

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) throw std::runtime_error(             \
        std::string("ASSERT_TRUE failed: ") + #cond); } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error( \
        std::string("ASSERT_NEAR failed: ") + #a + " (" +           \
        std::to_string(a) + ") vs " + #b + " (" +                   \
        std::to_string(b) + "), eps=" + std::to_string(eps)); } while (0)


// ═══════════════════════════════════════════════════════════════════════════
//  1. Event & EventQueue tests
// ═══════════════════════════════════════════════════════════════════════════

void test_event_type_to_string() {
    ASSERT_TRUE(event_type_to_string(EventType::REQUEST_ARRIVAL) == "REQUEST_ARRIVAL");
    ASSERT_TRUE(event_type_to_string(EventType::DECODE_COMPLETE) == "DECODE_COMPLETE");
    ASSERT_TRUE(event_type_to_string(EventType::CACHE_EVICTION) == "CACHE_EVICTION");
}

void test_event_ordering() {
    Event a{1.0, EventType::REQUEST_ARRIVAL, 0, 0};
    Event b{2.0, EventType::REQUEST_ARRIVAL, 1, 1};
    ASSERT_TRUE(a < b);   // earlier time first
    ASSERT_TRUE(!(b < a));
}

void test_event_ordering_tie_break() {
    Event a{1.0, EventType::REQUEST_ARRIVAL, 0, 0};
    Event b{1.0, EventType::PREFILL_COMPLETE, 1, 1};
    ASSERT_TRUE(a < b);   // same time → lower seq first
}

void test_event_queue_min_heap() {
    EventQueue q;
    q.push({3.0, EventType::REQUEST_ARRIVAL, 0});
    q.push({1.0, EventType::REQUEST_ARRIVAL, 1});
    q.push({2.0, EventType::REQUEST_ARRIVAL, 2});

    auto e1 = q.pop();
    auto e2 = q.pop();
    auto e3 = q.pop();

    ASSERT_NEAR(e1.time, 1.0, 1e-9);
    ASSERT_NEAR(e2.time, 2.0, 1e-9);
    ASSERT_NEAR(e3.time, 3.0, 1e-9);
    ASSERT_TRUE(q.empty());
}

void test_event_queue_insertion_order_preservation() {
    EventQueue q;
    // Same time, different request IDs
    q.push({1.0, EventType::REQUEST_ARRIVAL, 10});
    q.push({1.0, EventType::REQUEST_ARRIVAL, 20});
    q.push({1.0, EventType::REQUEST_ARRIVAL, 30});

    ASSERT_EQ(q.pop().request_id, 10);  // insertion order preserved
    ASSERT_EQ(q.pop().request_id, 20);
    ASSERT_EQ(q.pop().request_id, 30);
}


// ═══════════════════════════════════════════════════════════════════════════
//  2. Request lifecycle tests
// ═══════════════════════════════════════════════════════════════════════════

void test_request_initial_state() {
    Request r(1, 128, 64, 0.0);
    ASSERT_EQ(r.get_id(), 1);
    ASSERT_EQ(r.get_prompt_length(), 128);
    ASSERT_EQ(r.get_generation_length(), 64);
    ASSERT_EQ(r.get_tokens_generated(), 0);
    ASSERT_NEAR(r.get_arrival_time(), 0.0, 1e-9);
    ASSERT_TRUE(r.get_state() == RequestState::PENDING);
    ASSERT_TRUE(!r.is_completed());
}

void test_request_lifecycle_transitions() {
    Request r(1, 128, 64, 0.0);

    r.start_prefill(1.0);
    ASSERT_TRUE(r.get_state() == RequestState::IN_PREFILL);

    r.transition_to_decode(2.0);
    ASSERT_TRUE(r.get_state() == RequestState::IN_DECODE);
    ASSERT_NEAR(r.get_prefill_latency(), 1.0, 1e-9);  // 2.0 - 1.0

    r.complete(5.0);
    ASSERT_TRUE(r.get_state() == RequestState::COMPLETED);
    ASSERT_TRUE(r.is_completed());
    ASSERT_NEAR(r.get_decode_latency(), 3.0, 1e-9);    // 5.0 - 2.0
    ASSERT_NEAR(r.get_end_to_end_latency(), 5.0, 1e-9); // 5.0 - 0.0
}

void test_request_token_generation() {
    Request r(1, 10, 3, 0.0);
    ASSERT_TRUE(!r.generate_token());  // 1/3
    ASSERT_TRUE(!r.generate_token());  // 2/3
    ASSERT_TRUE(r.generate_token());   // 3/3 → done
    ASSERT_EQ(r.get_tokens_generated(), 3);
}


// ═══════════════════════════════════════════════════════════════════════════
//  3. KV Cache tests
// ═══════════════════════════════════════════════════════════════════════════

void test_kv_cache_basic() {
    KVCache cache(100, 2048);
    ASSERT_EQ(cache.current_size(), 0u);
    ASSERT_EQ(cache.capacity(), 100u);

    cache.generate_kv_cache(50, 0.0);
    ASSERT_EQ(cache.current_size(), 50u);
    ASSERT_EQ(cache.memory_bytes(), 50u * 2048u);
}

void test_kv_cache_hit_miss() {
    KVCache cache(100, 2048);
    cache.generate_kv_cache(10, 0.0);

    // Hit: access an existing token
    ASSERT_TRUE(cache.access_token(5, 1.0));
    ASSERT_EQ(cache.cache_hits(), 1);

    // Miss: access a non-existent token
    ASSERT_TRUE(!cache.access_token(999, 2.0));
    ASSERT_EQ(cache.cache_misses(), 1);
}

void test_kv_cache_hit_rate() {
    KVCache cache(100, 2048);
    cache.generate_kv_cache(10, 0.0);

    cache.access_token(0, 1.0);   // hit
    cache.access_token(1, 1.0);   // hit
    cache.access_token(999, 1.0); // miss

    ASSERT_NEAR(cache.hit_rate(), 2.0 / 3.0, 1e-6);
}

void test_kv_cache_eviction_random() {
    KVCache cache(10, 2048);
    cache.set_eviction_policy(std::make_unique<RandomEvictionPolicy>());

    // Insert 20 tokens into a cache with capacity 10
    cache.generate_kv_cache(20, 0.0);
    ASSERT_EQ(cache.current_size(), 10u);  // eviction brought it down
}

void test_kv_cache_eviction_lru() {
    KVCache cache(5, 2048);
    cache.set_eviction_policy(std::make_unique<LRUEvictionPolicy>());

    // Generate 5 tokens
    cache.generate_kv_cache(5, 0.0);
    ASSERT_EQ(cache.current_size(), 5u);

    // Access token 0 at a later time to make it "recently used"
    cache.access_token(0, 1.0);

    // Add a 6th token, triggering eviction
    cache.add_decode_token(100, 2.0);
    ASSERT_EQ(cache.current_size(), 5u);

    // Token 0 should still be present (it was recently accessed)
    ASSERT_TRUE(cache.access_token(0, 3.0));
}

void test_kv_cache_eviction_attention() {
    KVCache cache(5, 2048);
    cache.set_eviction_policy(std::make_unique<AttentionGuidedEvictionPolicy>());

    cache.generate_kv_cache(8, 0.0);
    // Should evict 3 entries with lowest attention scores
    ASSERT_EQ(cache.current_size(), 5u);
}

void test_kv_cache_clear() {
    KVCache cache(100, 2048);
    cache.generate_kv_cache(50, 0.0);
    ASSERT_EQ(cache.current_size(), 50u);

    cache.clear();
    ASSERT_EQ(cache.current_size(), 0u);
}


// ═══════════════════════════════════════════════════════════════════════════
//  4. Scheduler tests
// ═══════════════════════════════════════════════════════════════════════════

void test_fcfs_scheduler() {
    FirstComeFirstServeScheduler sched;
    ASSERT_TRUE(sched.policy_name() == "FCFS");

    std::vector<Request> pending = {
        Request(3, 100, 50, 3.0),
        Request(1, 100, 50, 1.0),
        Request(2, 100, 50, 2.0),
    };

    auto order = sched.schedule(pending);
    ASSERT_EQ(order.size(), 3u);
    ASSERT_EQ(order[0], 1);  // earliest arrival
    ASSERT_EQ(order[1], 2);
    ASSERT_EQ(order[2], 3);
}

void test_priority_scheduler() {
    PriorityScheduler sched(0.1, 0.5);
    ASSERT_TRUE(sched.policy_name() == "Priority");

    // Request A: score = 100*0.1 + 50*0.5 = 35
    // Request B: score = 10*0.1 + 10*0.5  = 6
    // Request C: score = 500*0.1 + 200*0.5 = 150
    std::vector<Request> pending = {
        Request(1, 100, 50, 0.0),   // score 35
        Request(2, 10, 10, 0.0),    // score 6  (highest priority)
        Request(3, 500, 200, 0.0),  // score 150 (lowest priority)
    };

    auto order = sched.schedule(pending);
    ASSERT_EQ(order[0], 2);  // lowest score → first
    ASSERT_EQ(order[1], 1);
    ASSERT_EQ(order[2], 3);
}

void test_continuous_batching_scheduler() {
    ContinuousBatchingScheduler sched(2);  // batch size = 2

    std::vector<Request> pending = {
        Request(1, 100, 50, 1.0),
        Request(2, 100, 50, 2.0),
        Request(3, 100, 50, 3.0),
    };

    auto order = sched.schedule(pending);
    ASSERT_EQ(order.size(), 2u);  // only 2 fit in the batch
    ASSERT_EQ(order[0], 1);
    ASSERT_EQ(order[1], 2);
}


// ═══════════════════════════════════════════════════════════════════════════
//  5. Hardware Profile tests
// ═══════════════════════════════════════════════════════════════════════════

void test_hardware_profiles() {
    auto h100 = H100_PROFILE();
    ASSERT_TRUE(h100.name == "H100_SXM");
    ASSERT_NEAR(h100.compute_tflops, 989.0, 1e-6);
    ASSERT_NEAR(h100.memory_capacity_gb, 80.0, 1e-6);

    auto a100 = A100_PROFILE();
    ASSERT_TRUE(a100.name == "A100_80GB");

    auto l4 = L4_PROFILE();
    ASSERT_TRUE(l4.name == "L4");
    ASSERT_NEAR(l4.memory_capacity_gb, 24.0, 1e-6);
}

void test_custom_profile() {
    auto custom = make_profile("Custom_GPU", 500.0, 2000.0, 48.0, 100.0, 0.05);
    ASSERT_TRUE(custom.name == "Custom_GPU");
    ASSERT_NEAR(custom.compute_tflops, 500.0, 1e-6);
}


// ═══════════════════════════════════════════════════════════════════════════
//  6. Engine Component tests
// ═══════════════════════════════════════════════════════════════════════════

void test_prefill_engine_availability() {
    PrefillEngine eng(H100_PROFILE(), 7.0);
    ASSERT_TRUE(eng.is_available());

    Request r(1, 100, 50, 0.0);
    eng.add_request(r, 0.0);
    ASSERT_TRUE(!eng.is_available());

    eng.release();
    ASSERT_TRUE(eng.is_available());
}

void test_prefill_engine_timing() {
    auto hw = H100_PROFILE();
    PrefillEngine eng(hw, 7.0);

    // time_per_token = 2 * 7.0 * 1e-3 / 989.0 ≈ 1.415e-5 s
    double expected_tpt = (2.0 * 7.0 * 1e-3) / 989.0;
    ASSERT_NEAR(eng.time_per_token(), expected_tpt, 1e-12);

    Request r(1, 100, 50, 0.0);
    eng.add_request(r, 0.0);
    double elapsed = eng.process_step(0.0);
    ASSERT_NEAR(elapsed, 100.0 * expected_tpt, 1e-9);
}

void test_decode_engine_timing() {
    auto hw = L4_PROFILE();
    DecodeEngine eng(hw, 7.0);

    // time_per_token = 2 * 7.0 / 300.0 ≈ 0.04667 s
    double expected_tpt = (2.0 * 7.0) / 300.0;
    ASSERT_NEAR(eng.time_per_token(), expected_tpt, 1e-9);

    Request r(1, 100, 10, 0.0);
    r.start_prefill(0.0);
    r.transition_to_decode(1.0);

    eng.add_request(r, 1.0);
    double dt = eng.time_per_token() + eng.active_requests().size() * 0.0001; // SimulationEngine logic
    ASSERT_NEAR(dt, expected_tpt + 0.0001, 1e-9);
}


// ═══════════════════════════════════════════════════════════════════════════
//  7. SimulationEngine integration tests
// ═══════════════════════════════════════════════════════════════════════════

void test_simulation_single_request() {
    auto sched = std::make_shared<FirstComeFirstServeScheduler>();
    SimulationEngine sim(sched, H100_PROFILE(), L4_PROFILE(), 7.0, 4096, 0.5);

    sim.add_request(1, 128, 32, 0.0);
    sim.run();

    ASSERT_EQ(sim.completed_count(), 1u);
    ASSERT_EQ(sim.total_request_count(), 1u);
    ASSERT_TRUE(sim.get_current_time() > 0.0);

    auto& s = sim.stats();
    ASSERT_EQ(s.total_requests_completed, 1);
    ASSERT_EQ(s.total_tokens_generated, 32);
    ASSERT_TRUE(s.total_prefill_time > 0.0);
    ASSERT_TRUE(s.total_decode_time > 0.0);
    ASSERT_TRUE(s.avg_e2e_latency() > 0.0);
    ASSERT_TRUE(s.throughput_tokens_per_sec() > 0.0);
}

void test_simulation_multiple_requests() {
    auto sched = std::make_shared<FirstComeFirstServeScheduler>();
    SimulationEngine sim(sched, H100_PROFILE(), L4_PROFILE(), 7.0, 4096, 0.5);

    sim.add_request(1, 64, 16, 0.0);
    sim.add_request(2, 128, 32, 0.1);
    sim.add_request(3, 256, 64, 0.2);

    sim.run();

    ASSERT_EQ(sim.completed_count(), 3u);
    ASSERT_EQ(sim.stats().total_requests_completed, 3);
    ASSERT_EQ(sim.stats().total_tokens_generated, 16 + 32 + 64);
}

void test_simulation_with_priority_scheduler() {
    auto sched = std::make_shared<PriorityScheduler>(0.1, 0.5);
    SimulationEngine sim(sched, H100_PROFILE(), L4_PROFILE(), 7.0, 4096, 0.5);

    // All arrive at time 0 — priority scheduler should pick the "cheapest" first
    sim.add_request(1, 1000, 200, 0.0);  // expensive
    sim.add_request(2, 10, 5, 0.0);      // cheap
    sim.add_request(3, 500, 100, 0.0);   // medium

    sim.run();
    ASSERT_EQ(sim.completed_count(), 3u);
}

void test_simulation_step_by_step() {
    auto sched = std::make_shared<FirstComeFirstServeScheduler>();
    SimulationEngine sim(sched, H100_PROFILE(), L4_PROFILE(), 7.0, 4096, 0.5);

    sim.add_request(1, 10, 5, 0.0);

    int steps = 0;
    while (sim.step()) {
        ++steps;
    }

    ASSERT_TRUE(steps > 0);
    ASSERT_EQ(sim.completed_count(), 1u);
}

void test_simulation_add_request_obj() {
    auto sched = std::make_shared<FirstComeFirstServeScheduler>();
    SimulationEngine sim(sched, H100_PROFILE(), L4_PROFILE());

    Request r(42, 100, 20, 0.0);
    sim.add_request_obj(r);
    sim.run();

    ASSERT_EQ(sim.completed_count(), 1u);
    auto completed = sim.get_completed_requests();
    ASSERT_EQ(completed.size(), 1u);
    ASSERT_EQ(completed[0].get_id(), 42);
}

void test_simulation_eviction_policy() {
    auto sched = std::make_shared<FirstComeFirstServeScheduler>();
    // Small KV cache capacity to force evictions
    SimulationEngine sim(sched, H100_PROFILE(), L4_PROFILE(), 7.0, 50, 0.5);
    sim.set_eviction_policy_for_all("LRU");

    sim.add_request(1, 100, 20, 0.0);  // 100 tokens > 50 capacity → eviction
    sim.run();

    ASSERT_EQ(sim.completed_count(), 1u);
}

void test_simulation_scheduler_name() {
    auto sched = std::make_shared<ContinuousBatchingScheduler>(4);
    SimulationEngine sim(sched, A100_PROFILE(), A100_PROFILE());
    ASSERT_TRUE(sim.scheduler_name() == "ContinuousBatching(batch=4)");
}


// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    std::cout << "\n=== LLM Sim Bench — C++ Unit Tests ===\n\n";

    std::cout << "[Event & EventQueue]\n";
    RUN_TEST(test_event_type_to_string);
    RUN_TEST(test_event_ordering);
    RUN_TEST(test_event_ordering_tie_break);
    RUN_TEST(test_event_queue_min_heap);
    RUN_TEST(test_event_queue_insertion_order_preservation);

    std::cout << "\n[Request Lifecycle]\n";
    RUN_TEST(test_request_initial_state);
    RUN_TEST(test_request_lifecycle_transitions);
    RUN_TEST(test_request_token_generation);

    std::cout << "\n[KV Cache]\n";
    RUN_TEST(test_kv_cache_basic);
    RUN_TEST(test_kv_cache_hit_miss);
    RUN_TEST(test_kv_cache_hit_rate);
    RUN_TEST(test_kv_cache_eviction_random);
    RUN_TEST(test_kv_cache_eviction_lru);
    RUN_TEST(test_kv_cache_eviction_attention);
    RUN_TEST(test_kv_cache_clear);

    std::cout << "\n[Schedulers]\n";
    RUN_TEST(test_fcfs_scheduler);
    RUN_TEST(test_priority_scheduler);
    RUN_TEST(test_continuous_batching_scheduler);

    std::cout << "\n[Hardware Profiles]\n";
    RUN_TEST(test_hardware_profiles);
    RUN_TEST(test_custom_profile);

    std::cout << "\n[Engine Components]\n";
    RUN_TEST(test_prefill_engine_availability);
    RUN_TEST(test_prefill_engine_timing);
    RUN_TEST(test_decode_engine_timing);

    std::cout << "\n[SimulationEngine Integration]\n";
    RUN_TEST(test_simulation_single_request);
    RUN_TEST(test_simulation_multiple_requests);
    RUN_TEST(test_simulation_with_priority_scheduler);
    RUN_TEST(test_simulation_step_by_step);
    RUN_TEST(test_simulation_add_request_obj);
    RUN_TEST(test_simulation_eviction_policy);
    RUN_TEST(test_simulation_scheduler_name);

    std::cout << "\n══════════════════════════════════════\n";
    std::cout << "  Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n";
    std::cout << "══════════════════════════════════════\n\n";

    return tests_failed > 0 ? 1 : 0;
}
