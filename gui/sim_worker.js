/**
 * sim_worker.js — Web Worker: Faithful DES engine with Agentic LLM Traffic Modeling.
 *
 * Models 6 agentic traffic classes with distinct compute/memory profiles:
 *   1. SimpleQuery    — Short Q&A, chatbot-style (low compute, low bandwidth)
 *   2. ToolCall       — Function/API invocation (short prompt, tiny gen)
 *   3. MultiStepCoT   — Chain-of-Thought reasoning (long prompt, long gen)
 *   4. RecursiveLoop  — Iterative refinement / self-correction (medium prompt, very long gen)
 *   5. MultiRAG       — Retrieval-Augmented Generation (very long prompt from docs, medium gen)
 *   6. CodeGen        — Code generation / completion (medium prompt, long gen)
 *
 * Implements:
 *  - Min-heap event queue with tie-breaking sequence numbers
 *  - PrefillEngine (compute-bound, single request)
 *  - DecodeEngine (bandwidth-bound, batched requests)
 *  - KVCache with pluggable eviction (None / Random / LRU / AttentionGuided)
 *  - Scheduler-aware decode queue (FCFS / Priority / ContinuousBatching)
 *  - Cache-miss recompute penalties on every decode step
 *  - Queue depth timeline tracking for visualization
 */

// ═══════════════════════════════════════════════════════════════════════════════
//  Agentic Traffic Type Definitions
// ═══════════════════════════════════════════════════════════════════════════════

const TRAFFIC_TYPES = {
    SimpleQuery: {
        name: 'SimpleQuery',
        label: 'Simple Query',
        color: '#22d3ee',  // cyan
        promptRange: [32, 256],
        genRange: [16, 128],
        weight: 0.25,  // 25% of agentic traffic
        burstiness: 1.0,  // uniform arrival
    },
    ToolCall: {
        name: 'ToolCall',
        label: 'Tool Call',
        color: '#f59e0b',  // amber
        promptRange: [64, 512],
        genRange: [4, 32],   // very short — just a function call output
        weight: 0.20,
        burstiness: 3.0,  // bursty — tool calls come in rapid sequences
    },
    MultiStepCoT: {
        name: 'MultiStepCoT',
        label: 'Multi-Step CoT',
        color: '#a78bfa',  // violet
        promptRange: [512, 4096],
        genRange: [256, 1024],
        weight: 0.15,
        burstiness: 0.5,  // slower arrival — expensive reasoning tasks
    },
    RecursiveLoop: {
        name: 'RecursiveLoop',
        label: 'Recursive Loop',
        color: '#ef4444',  // red
        promptRange: [128, 1024],
        genRange: [512, 2048],  // very long — iterative self-correction
        weight: 0.10,
        burstiness: 0.3,  // rare but expensive
    },
    MultiRAG: {
        name: 'MultiRAG',
        label: 'Multi-RAG Query',
        color: '#10b981',  // emerald
        promptRange: [2048, 8192],  // very long from retrieved docs
        genRange: [64, 512],
        weight: 0.15,
        burstiness: 1.5,
    },
    CodeGen: {
        name: 'CodeGen',
        label: 'Code Generation',
        color: '#3b82f6',  // blue
        promptRange: [256, 2048],
        genRange: [128, 1024],
        weight: 0.15,
        burstiness: 0.8,
    },
};

const TRAFFIC_TYPE_NAMES = Object.keys(TRAFFIC_TYPES);

// ═══════════════════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════════════════

const EventType = {
    REQUEST_ARRIVAL: 0,
    PREFILL_COMPLETE: 1,
    KV_TRANSFER_COMPLETE: 2,
    DECODE_STEP_COMPLETE: 3,
};

const RequestState = {
    PENDING: 'PENDING',
    IN_PREFILL: 'IN_PREFILL',
    IN_DECODE: 'IN_DECODE',
    COMPLETED: 'COMPLETED',
};

// ═══════════════════════════════════════════════════════════════════════════════
//  Min-Heap Event Queue
// ═══════════════════════════════════════════════════════════════════════════════

class EventQueue {
    constructor() { this.heap = []; this.nextSeq = 0; }
    push(event) {
        event.seq = this.nextSeq++;
        this.heap.push(event);
        this._bubbleUp(this.heap.length - 1);
    }
    pop() {
        const top = this.heap[0];
        const last = this.heap.pop();
        if (this.heap.length > 0) { this.heap[0] = last; this._sinkDown(0); }
        return top;
    }
    empty() { return this.heap.length === 0; }
    size() { return this.heap.length; }
    _compare(a, b) { return a.time !== b.time ? a.time - b.time : a.seq - b.seq; }
    _bubbleUp(i) {
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (this._compare(this.heap[i], this.heap[p]) < 0) {
                [this.heap[i], this.heap[p]] = [this.heap[p], this.heap[i]]; i = p;
            } else break;
        }
    }
    _sinkDown(i) {
        const n = this.heap.length;
        while (true) {
            let s = i; const l = 2*i+1, r = 2*i+2;
            if (l < n && this._compare(this.heap[l], this.heap[s]) < 0) s = l;
            if (r < n && this._compare(this.heap[r], this.heap[s]) < 0) s = r;
            if (s !== i) { [this.heap[i], this.heap[s]] = [this.heap[s], this.heap[i]]; i = s; } else break;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Request (with traffic type tagging)
// ═══════════════════════════════════════════════════════════════════════════════

class Request {
    constructor(id, promptLen, genLen, arrivalTime, trafficType = 'SimpleQuery') {
        this.id = id;
        this.promptLength = promptLen;
        this.generationLength = genLen;
        this.arrivalTime = arrivalTime;
        this.trafficType = trafficType;
        this.state = RequestState.PENDING;
        this.tokensGenerated = 0;
        this.prefillStart = 0;
        this.prefillEnd = 0;
        this.decodeStart = 0;
        this.completionTime = 0;
    }
    startPrefill(t) { this.state = RequestState.IN_PREFILL; this.prefillStart = t; }
    transitionToDecode(t) { this.state = RequestState.IN_DECODE; this.prefillEnd = t; this.decodeStart = t; }
    complete(t) { this.state = RequestState.COMPLETED; this.completionTime = t; }
    generateToken() { this.tokensGenerated++; return this.tokensGenerated >= this.generationLength; }
    get prefillLatency() { return this.prefillEnd - this.prefillStart; }
    get decodeLatency() { return this.completionTime - this.decodeStart; }
    get e2eLatency() { return this.completionTime - this.arrivalTime; }
    get isCompleted() { return this.state === RequestState.COMPLETED; }

    // Roofline: operational intensity = FLOPs / bytes accessed
    // Prefill is compute-bound: 2*P*N FLOPs for N tokens, P params
    // Decode is memory-bound: 2*P bytes per token (weight loading)
    get operationalIntensity() {
        // FLOPs for this request: prefill (2*P*promptLen) + decode (2*P*genLen)
        // Bytes accessed: prefill (2*P once) + decode (2*P * genLen loads)
        const totalFlops = 2 * this.promptLength + 2 * this.generationLength;
        const totalBytes = 2 + 2 * this.generationLength; // normalized by P
        return totalFlops / totalBytes; // FLOP/byte ratio
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  KV Cache with Eviction Policies
// ═══════════════════════════════════════════════════════════════════════════════

class KVCache {
    constructor(capacity = 4096) {
        this.capacity = capacity;
        this.entries = new Map();
        this.hits = 0;
        this.misses = 0;
        this.evictions = 0;
        this.evictionPolicy = 'None';
    }

    generateKvCache(numTokens, time) {
        this.entries.clear();
        for (let i = 0; i < numTokens; i++) {
            this.entries.set(i, { lastAccess: time, score: Math.random() * 0.5 + 0.5 });
        }
        this._evictIfNeeded();
    }

    accessToken(idx, time) {
        if (this.entries.has(idx)) {
            this.hits++;
            this.entries.get(idx).lastAccess = time;
            return true;
        }
        this.misses++;
        this.entries.set(idx, { lastAccess: time, score: Math.random() * 0.3 + 0.1 });
        return false;
    }

    addDecodeToken(idx, time) {
        this.entries.set(idx, { lastAccess: time, score: Math.random() * 0.3 + 0.1 });
        this._evictIfNeeded();
    }

    evictToCapacity() {
        this._evictIfNeeded();
    }

    _evictIfNeeded() {
        if (this.entries.size <= this.capacity) return;
        const numToEvict = this.entries.size - this.capacity;
        this.evictions += numToEvict;

        if (numToEvict === 1) {
            if (this.evictionPolicy === 'LRU') {
                let oldest = null, oldestKey = -1;
                for (const [k, v] of this.entries) {
                    if (oldest === null || v.lastAccess < oldest) { oldest = v.lastAccess; oldestKey = k; }
                }
                if (oldestKey >= 0) this.entries.delete(oldestKey);
            } else if (this.evictionPolicy === 'AttentionGuided') {
                let lowestScore = Infinity, lowestKey = -1;
                for (const [k, v] of this.entries) {
                    if (v.score < lowestScore) { lowestScore = v.score; lowestKey = k; }
                }
                if (lowestKey >= 0) this.entries.delete(lowestKey);
            } else if (this.evictionPolicy === 'Random') {
                const keys = Array.from(this.entries.keys());
                this.entries.delete(keys[Math.floor(Math.random() * keys.length)]);
            } else {
                this.entries.delete(this.entries.keys().next().value);
            }
            return;
        }

        if (this.evictionPolicy === 'LRU') {
            const sorted = Array.from(this.entries.entries())
                .sort((a, b) => a[1].lastAccess - b[1].lastAccess);
            for (let i = 0; i < numToEvict; i++) {
                this.entries.delete(sorted[i][0]);
            }
        } else if (this.evictionPolicy === 'AttentionGuided') {
            const sorted = Array.from(this.entries.entries())
                .sort((a, b) => a[1].score - b[1].score);
            for (let i = 0; i < numToEvict; i++) {
                this.entries.delete(sorted[i][0]);
            }
        } else if (this.evictionPolicy === 'Random') {
            const keys = Array.from(this.entries.keys());
            for (let i = 0; i < numToEvict; i++) {
                const idx = Math.floor(Math.random() * keys.length);
                this.entries.delete(keys[idx]);
                keys.splice(idx, 1);
            }
        } else {
            const keys = this.entries.keys();
            for (let i = 0; i < numToEvict; i++) {
                this.entries.delete(keys.next().value);
            }
        }
    }

    get currentSize() { return this.entries.size; }
    get hitRate() { const t = this.hits + this.misses; return t > 0 ? this.hits / t : 0; }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Hardware Profiles
// ═══════════════════════════════════════════════════════════════════════════════

const PROFILES = {
    H100_SXM:  { name: 'H100_SXM',  computeTflops: 989,  memBwGbps: 3350, memCapGb: 80  },
    A100_80GB: { name: 'A100_80GB', computeTflops: 312,  memBwGbps: 2039, memCapGb: 80  },
    L4:        { name: 'L4',        computeTflops: 121,  memBwGbps: 300,  memCapGb: 24  },
};

// ═══════════════════════════════════════════════════════════════════════════════
//  K-Means Clustering (for traffic classification)
// ═══════════════════════════════════════════════════════════════════════════════

function kMeansClustering(points, k = 4, maxIter = 50) {
    if (points.length < k) return points.map((_, i) => i % k);

    // Normalize features
    const dims = points[0].length;
    const mins = new Array(dims).fill(Infinity);
    const maxs = new Array(dims).fill(-Infinity);
    for (const p of points) {
        for (let d = 0; d < dims; d++) {
            mins[d] = Math.min(mins[d], p[d]);
            maxs[d] = Math.max(maxs[d], p[d]);
        }
    }
    const normalized = points.map(p => p.map((v, d) => {
        const range = maxs[d] - mins[d];
        return range > 0 ? (v - mins[d]) / range : 0;
    }));

    // K-means++ initialization
    const centroids = [normalized[Math.floor(Math.random() * normalized.length)]];
    while (centroids.length < k) {
        const dists = normalized.map(p => {
            let minD = Infinity;
            for (const c of centroids) {
                let d = 0;
                for (let i = 0; i < dims; i++) d += (p[i] - c[i]) ** 2;
                minD = Math.min(minD, d);
            }
            return minD;
        });
        const totalDist = dists.reduce((a, b) => a + b, 0);
        let r = Math.random() * totalDist, cumul = 0;
        for (let i = 0; i < normalized.length; i++) {
            cumul += dists[i];
            if (cumul >= r) { centroids.push([...normalized[i]]); break; }
        }
    }

    // Iterate
    let assignments = new Array(normalized.length).fill(0);
    for (let iter = 0; iter < maxIter; iter++) {
        // Assign
        for (let i = 0; i < normalized.length; i++) {
            let bestCluster = 0, bestDist = Infinity;
            for (let c = 0; c < k; c++) {
                let d = 0;
                for (let dim = 0; dim < dims; dim++) d += (normalized[i][dim] - centroids[c][dim]) ** 2;
                if (d < bestDist) { bestDist = d; bestCluster = c; }
            }
            assignments[i] = bestCluster;
        }
        // Update centroids
        for (let c = 0; c < k; c++) {
            const members = normalized.filter((_, i) => assignments[i] === c);
            if (members.length > 0) {
                centroids[c] = new Array(dims).fill(0);
                for (const m of members) for (let d = 0; d < dims; d++) centroids[c][d] += m[d];
                for (let d = 0; d < dims; d++) centroids[c][d] /= members.length;
            }
        }
    }
    return assignments;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  SimulationEngine
// ═══════════════════════════════════════════════════════════════════════════════

class SimulationEngine {
    constructor(config) {
        this.config = config;
        this.currentTime = 0;
        this.eventQueue = new EventQueue();
        this.requests = new Map();
        this.kvCaches = new Map();
        this.pendingIds = [];
        this.decodeWaiting = [];

        this.prefillBusy = false;
        this.prefillReqId = -1;
        this.activeDecodeRequests = [];
        this.decodeStepScheduled = false;

        const prefillHw = PROFILES[config.prefillHw] || PROFILES.H100_SXM;
        const decodeHw = PROFILES[config.decodeHw] || PROFILES.L4;
        const params = config.modelParamsB || 7.0;
        this.prefillTimePerToken = (2.0 * params * 1e-3) / prefillHw.computeTflops;
        this.decodeTimePerToken = (2.0 * params) / decodeHw.memBwGbps;
        this.kvTransferLatency = (config.kvTransferLatencyMs || 0.5) / 1000.0;
        this.modelParamsB = params;
        this.prefillHw = prefillHw;
        this.decodeHw = decodeHw;

        this.stats = {
            totalCompleted: 0, totalTokens: 0,
            totalPrefillTime: 0, totalDecodeTime: 0,
            totalE2eLatency: 0, maxE2eLatency: 0,
            totalCacheHits: 0, totalCacheMisses: 0, totalEvictions: 0,
            peakQueueDepth: 0,
        };

        this.latencies = [];
        this.queueTimeline = [];  // {time, pending, decodeWaiting, activeDecode}
        this._lastQueueSnapshot = -1;
        this._snapshotInterval = 0.5;
    }

    addRequest(id, promptLen, genLen, arrivalTime, trafficType = 'SimpleQuery') {
        const req = new Request(id, promptLen, genLen, arrivalTime, trafficType);
        this.requests.set(id, req);
        const cache = new KVCache(this.config.kvCacheCapacity || 4096);
        cache.evictionPolicy = this.config.evictionPolicy || 'None';
        this.kvCaches.set(id, cache);
        this.eventQueue.push({ time: arrivalTime, type: EventType.REQUEST_ARRIVAL, requestId: id });
    }

    step() {
        if (this.eventQueue.empty()) return false;
        const ev = this.eventQueue.pop();
        this.currentTime = ev.time;

        switch (ev.type) {
            case EventType.REQUEST_ARRIVAL:      this._handleArrival(ev); break;
            case EventType.PREFILL_COMPLETE:     this._handlePrefillComplete(ev); break;
            case EventType.KV_TRANSFER_COMPLETE: this._handleKvTransfer(ev); break;
            case EventType.DECODE_STEP_COMPLETE: this._handleDecodeStep(ev); break;
        }

        // Sample queue depth, but prevent unbound array growth
        if (this.currentTime - this._lastQueueSnapshot >= this._snapshotInterval) {
            this.queueTimeline.push({
                time: this.currentTime,
                pending: this.pendingIds.length,
                decodeWaiting: this.decodeWaiting.length,
                activeDecode: this.activeDecodeRequests.length,
            });
            this._lastQueueSnapshot = this.currentTime;
            
            if (this.queueTimeline.length > 500) {
                this.queueTimeline = this.queueTimeline.filter((_, i) => i % 2 === 0);
                this._snapshotInterval *= 2;
            }
        }

        return true;
    }

    run() { while (this.step()) {} }

    _scheduleOrder(pendingReqs) {
        const policy = this.config.scheduler || 'FCFS';
        const sorted = [...pendingReqs];
        if (policy === 'Priority') {
            const pw = this.config.priorityPromptWeight || 0.1;
            const gw = this.config.priorityGenWeight || 0.5;
            sorted.sort((a, b) => {
                return (a.promptLength * pw + a.generationLength * gw) -
                       (b.promptLength * pw + b.generationLength * gw);
            });
        } else {
            sorted.sort((a, b) => a.arrivalTime - b.arrivalTime);
        }
        if (policy === 'ContinuousBatching') {
            return sorted.slice(0, this.config.maxBatchSize || 8).map(r => r.id);
        }
        return sorted.map(r => r.id);
    }

    _trySchedulePrefill() {
        if (this.pendingIds.length === 0 || this.prefillBusy) return;
        const pendingReqs = this.pendingIds.map(id => this.requests.get(id));
        const ordered = this._scheduleOrder(pendingReqs);
        if (ordered.length === 0) return;
        const chosenId = ordered[0];
        this.pendingIds = this.pendingIds.filter(id => id !== chosenId);
        const req = this.requests.get(chosenId);
        this.prefillBusy = true;
        this.prefillReqId = chosenId;
        req.startPrefill(this.currentTime);
        const dt = req.promptLength * this.prefillTimePerToken;
        this.eventQueue.push({ time: this.currentTime + dt, type: EventType.PREFILL_COMPLETE, requestId: chosenId });
    }

    _tryScheduleDecode() {
        if (this.decodeWaiting.length > 0) {
            const isCB = (this.config.scheduler === 'ContinuousBatching');
            if (!isCB && this.activeDecodeRequests.length > 0) {
                // sequential
            } else {
                const pendingReqs = this.decodeWaiting.map(id => this.requests.get(id));
                const ordered = this._scheduleOrder(pendingReqs);
                if (ordered.length > 0) {
                    const maxBatch = this.config.maxBatchSize || 8;
                    for (const id of ordered) {
                        if (!isCB && this.activeDecodeRequests.length > 0) break;
                        if (isCB && this.activeDecodeRequests.length >= maxBatch) break;
                        const req = this.requests.get(id);
                        req.transitionToDecode(this.currentTime);
                        this.activeDecodeRequests.push(id);
                        this.decodeWaiting = this.decodeWaiting.filter(wid => wid !== id);
                    }
                }
            }
        }

        if (this.activeDecodeRequests.length > 0 && !this.decodeStepScheduled) {
            const batchSize = this.activeDecodeRequests.length;
            let baseDt = this.decodeTimePerToken + batchSize * 0.0001;
            let maxMissPenalty = 0;

            for (const reqId of this.activeDecodeRequests) {
                const req = this.requests.get(reqId);
                const cache = this.kvCaches.get(reqId);
                const totalTokensToAttend = req.promptLength + req.tokensGenerated;
                let misses = 0;
                for (let i = 0; i < totalTokensToAttend; i++) {
                    if (!cache.accessToken(i, this.currentTime)) misses++;
                }
                cache.entries.set(totalTokensToAttend, { lastAccess: this.currentTime, score: Math.random() * 0.3 + 0.1 });
                cache.evictToCapacity();
                maxMissPenalty = Math.max(maxMissPenalty, misses * this.prefillTimePerToken);
            }

            this.eventQueue.push({ time: this.currentTime + baseDt + maxMissPenalty, type: EventType.DECODE_STEP_COMPLETE, requestId: -1 });
            this.decodeStepScheduled = true;
        }
    }

    _handleArrival(ev) {
        this.pendingIds.push(ev.requestId);
        this.stats.peakQueueDepth = Math.max(this.stats.peakQueueDepth, this.pendingIds.length);
        this._trySchedulePrefill();
    }

    _handlePrefillComplete(ev) {
        const req = this.requests.get(ev.requestId);
        const cache = this.kvCaches.get(ev.requestId);
        cache.generateKvCache(req.promptLength, this.currentTime);
        this.stats.totalCacheHits += cache.hits;
        this.stats.totalCacheMisses += cache.misses;
        this.prefillBusy = false;
        this.prefillReqId = -1;
        this.eventQueue.push({
            time: this.currentTime + this.kvTransferLatency,
            type: EventType.KV_TRANSFER_COMPLETE, requestId: ev.requestId,
        });
        this._trySchedulePrefill();
    }

    _handleKvTransfer(ev) {
        this.decodeWaiting.push(ev.requestId);
        this._tryScheduleDecode();
    }

    _handleDecodeStep(ev) {
        this.decodeStepScheduled = false;
        const currentBatch = [...this.activeDecodeRequests];

        for (const reqId of currentBatch) {
            const req = this.requests.get(reqId);
            const done = req.generateToken();
            this.stats.totalTokens++;

            if (done) {
                req.complete(this.currentTime);
                this.activeDecodeRequests = this.activeDecodeRequests.filter(id => id !== reqId);
                this.stats.totalCompleted++;
                this.stats.totalPrefillTime += req.prefillLatency;
                this.stats.totalDecodeTime += req.decodeLatency;
                const e2e = req.e2eLatency;
                this.stats.totalE2eLatency += e2e;
                this.stats.maxE2eLatency = Math.max(this.stats.maxE2eLatency, e2e);
                const cache = this.kvCaches.get(reqId);
                this.stats.totalCacheHits += cache.hits;
                this.stats.totalCacheMisses += cache.misses;
                this.stats.totalEvictions += cache.evictions;

                // Compute per-request throughput (tokens / decode_time)
                const reqThroughput = req.decodeLatency > 0 ? req.generationLength / req.decodeLatency : 0;

                this.latencies.push({
                    id: req.id, e2e: req.e2eLatency,
                    prefill: req.prefillLatency, decode: req.decodeLatency,
                    prompt: req.promptLength, gen: req.generationLength,
                    arrival: req.arrivalTime, completion: req.completionTime,
                    trafficType: req.trafficType,
                    operationalIntensity: req.operationalIntensity,
                    reqThroughput: reqThroughput,
                });
            }
        }
        this._tryScheduleDecode();
    }

    snapshot() {
        const completed = this.stats.totalCompleted;
        const avgLat = completed > 0 ? this.stats.totalE2eLatency / completed : 0;
        const tput = this.currentTime > 0 ? this.stats.totalTokens / this.currentTime : 0;
        const totalHM = this.stats.totalCacheHits + this.stats.totalCacheMisses;
        const hitRate = totalHM > 0 ? this.stats.totalCacheHits / totalHM : 0;

        let p50 = 0, p95 = 0, p99 = 0;
        if (this.latencies.length > 0) {
            const sorted = this.latencies.map(l => l.e2e).sort((a, b) => a - b);
            p50 = sorted[Math.floor(sorted.length * 0.50)] || 0;
            p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;
            p99 = sorted[Math.floor(sorted.length * 0.99)] || 0;
        }

        // Traffic mix counts
        const trafficMix = {};
        for (const ttype of TRAFFIC_TYPE_NAMES) trafficMix[ttype] = 0;
        for (const [, req] of this.requests) trafficMix[req.trafficType] = (trafficMix[req.trafficType] || 0) + 1;

        // K-means clustering on completed requests (4 hardware-bound profiles)
        let clusterAssignments = [];
        if (this.latencies.length >= 4) {
            const maxSamples = 500;
            const step = Math.max(1, Math.floor(this.latencies.length / maxSamples));
            const samples = [];
            for (let i = 0; i < this.latencies.length; i += step) {
                const l = this.latencies[i];
                samples.push([l.prompt, l.gen, l.e2e, l.operationalIntensity]);
            }
            
            clusterAssignments = kMeansClustering(samples, 4);
            
            // Map centroids back to all points
            const centroids = [];
            for (let c = 0; c < 4; c++) {
                const members = samples.filter((_, i) => clusterAssignments[i] === c);
                if (members.length > 0) {
                    centroids.push([
                        members.reduce((sum, m) => sum + m[0], 0) / members.length,
                        members.reduce((sum, m) => sum + m[1], 0) / members.length,
                        members.reduce((sum, m) => sum + m[2], 0) / members.length,
                        members.reduce((sum, m) => sum + m[3], 0) / members.length
                    ]);
                } else {
                    centroids.push([0, 0, 0, 0]);
                }
            }
            
            // Re-assign all latencies based on Euclidean distance to centroids
            // We must normalize first because kMeansClustering normalizes internally, but here we do a rough distance
            // Since kMeansClustering normalized before clustering, doing raw distance here might be slightly skewed,
            // but it's fast enough and good enough for the UI.
            clusterAssignments = this.latencies.map(l => {
                let bestD = Infinity, bestC = 0;
                for (let c = 0; c < 4; c++) {
                    const d = Math.pow(l.prompt - centroids[c][0], 2) +
                              Math.pow(l.gen - centroids[c][1], 2) +
                              Math.pow(l.e2e - centroids[c][2], 2) +
                              Math.pow(l.operationalIntensity - centroids[c][3], 2);
                    if (d < bestD) { bestD = d; bestC = c; }
                }
                return bestC;
            });
        }

        return {
            time: this.currentTime, completed, total: this.requests.size,
            pending: this.pendingIds.length,
            prefillBusy: this.prefillBusy,
            decodeBusy: this.activeDecodeRequests.length > 0,
            decodeBatchSize: this.activeDecodeRequests.length,
            avgLatency: avgLat, maxLatency: this.stats.maxE2eLatency,
            p50, p95, p99,
            throughput: tput, totalTokens: this.stats.totalTokens,
            cacheHitRate: hitRate,
            cacheHits: this.stats.totalCacheHits, cacheMisses: this.stats.totalCacheMisses,
            evictions: this.stats.totalEvictions,
            queueDepth: this.pendingIds.length, peakQueueDepth: this.stats.peakQueueDepth,
            decodeWaiting: this.decodeWaiting.length,
            latencies: this.latencies,
            queueTimeline: this.queueTimeline,
            trafficMix, clusterAssignments,
            trafficTypes: TRAFFIC_TYPES,
            scheduler: this.config.scheduler || 'FCFS',
            evictionPolicy: this.config.evictionPolicy || 'None',
            // Roofline bounds
            peakComputeTput: this.prefillHw.computeTflops * 1e12 / (2 * this.modelParamsB * 1e9),
            peakMemBwTput: this.decodeHw.memBwGbps * 1e9 / (2 * this.modelParamsB * 1e9),
        };
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Worker Message Handler
// ═══════════════════════════════════════════════════════════════════════════════

let engine = null;
let running = false;
let stepBatch = 50;

self.onmessage = function(e) {
    const { type, payload } = e.data;

    switch (type) {
        case 'INIT': {
            engine = new SimulationEngine(payload.config);
            const wl = payload.workload;
            const rng = mulberry32(wl.seed || 42);
            let arrivalTime = 0;
            const isAgentic = payload.config.trafficProfile === 'Agentic';

            // Helper for M/G/c heavy-tail sampling
            const samplePareto = (minVal, maxVal, alpha) => {
                const u = rng();
                const val = minVal / Math.pow(1 - u, 1 / alpha);
                return Math.floor(Math.min(val, maxVal));
            };

            for (let i = 0; i < wl.numRequests; i++) {
                let promptLen, genLen, trafficType;

                if (isAgentic) {
                    // Sample a traffic type based on weights
                    trafficType = sampleTrafficType(rng);
                    const tt = TRAFFIC_TYPES[trafficType];
                    // Agentic traffic service times are heavy-tailed (M/G/c queues)
                    promptLen = samplePareto(tt.promptRange[0], tt.promptRange[1], 1.5);
                    genLen = samplePareto(tt.genRange[0], tt.genRange[1], 1.2);
                    // Bursty arrival: scale inter-arrival by burstiness factor
                    arrivalTime += -Math.log(1 - rng()) / (wl.arrivalRate * tt.burstiness);
                } else {
                    trafficType = 'SimpleQuery';
                    // Standard LLM traffic uses Uniform/M/M/c-like behavior
                    promptLen = Math.floor(rng() * (wl.promptMax - wl.promptMin + 1)) + wl.promptMin;
                    genLen = Math.floor(rng() * (wl.genMax - wl.genMin + 1)) + wl.genMin;
                    arrivalTime += -Math.log(1 - rng()) / wl.arrivalRate;
                }

                engine.addRequest(i + 1, promptLen, genLen, arrivalTime, trafficType);
            }

            self.postMessage({ type: 'READY', payload: engine.snapshot() });
            break;
        }

        case 'START': {
            running = true;
            stepBatch = payload?.stepBatch || 50;
            runLoop();
            break;
        }

        case 'PAUSE': { running = false; break; }

        case 'STEP': {
            if (engine) {
                const hasMore = engine.step();
                self.postMessage({ type: 'UPDATE', payload: engine.snapshot() });
                if (!hasMore) self.postMessage({ type: 'DONE', payload: engine.snapshot() });
            }
            break;
        }

        case 'RUN_TO_COMPLETION': {
            if (engine) {
                engine.run();
                self.postMessage({ type: 'DONE', payload: engine.snapshot() });
            }
            break;
        }

        case 'SET_STEP_BATCH': { stepBatch = payload.stepBatch; break; }
    }
};

function sampleTrafficType(rng) {
    const r = rng();
    let cumul = 0;
    for (const name of TRAFFIC_TYPE_NAMES) {
        cumul += TRAFFIC_TYPES[name].weight;
        if (r <= cumul) return name;
    }
    return TRAFFIC_TYPE_NAMES[TRAFFIC_TYPE_NAMES.length - 1];
}

function runLoop() {
    if (!running || !engine) return;
    const startTime = performance.now();
    let steps = 0;
    while (performance.now() - startTime < 14) { // 14ms time budget per frame
        if (!engine.step()) {
            running = false;
            self.postMessage({ type: 'DONE', payload: engine.snapshot() });
            return;
        }
        steps++;
    }
    self.postMessage({ type: 'UPDATE', payload: engine.snapshot() });
    setTimeout(runLoop, 0); // Yield to JS event loop
}

function mulberry32(a) {
    return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
