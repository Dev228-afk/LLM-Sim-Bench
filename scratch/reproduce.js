const fs = require('fs');
const vm = require('vm');

const workerCode = fs.readFileSync('./gui/sim_worker.js', 'utf8') + '\nself.SimulationEngine = SimulationEngine;';

const PROFILES = {
    models: {
        'Llama-3-70B': { paramsB: 70, layers: 80, kvHeads: 8, headDim: 128 }
    },
    'GB300': { name: 'GB300 288GB', vramGb: 288, tflops: 3000, memBwGbps: 10000 }
};

const sandbox = { PROFILES, console, Math, self: {} };
vm.createContext(sandbox);
vm.runInContext(workerCode, sandbox);

const config = {
    useManualModel: false,
    modelParamsB: 70,
    modelLayers: 80,
    modelKvHeads: 8,
    modelHeadDim: 128,
    maxModelLen: 8192,
    tensorParallelSize: 7,
    prefillHw: 'GB300',
    prefillChunkSize: 0,
    decodeHw: 'GB300',
    gpuMemoryUtilization: 0.90,
    scheduler: 'PriorityContinuousBatching',
    priorityPromptWeight: 0.1,
    priorityGenWeight: 0.5,
    preemptionThreshold: 0.5,
    maxNumBatchedTokens: 4096,
    maxBatchSize: 8,
    kvCacheCapacity: 4096
};

const engine = new sandbox.self.SimulationEngine(config);

const numRequests = 50;
const arrivalRate = 10;
const promptMin = 64;
const promptMax = 2048;
const genMin = 16;
const genMax = 512;

let arrivalTime = 0.0;
for (let i = 0; i < numRequests; i++) {
    const promptLen = Math.floor(Math.random() * (promptMax - promptMin + 1)) + promptMin;
    const genLen = Math.floor(Math.random() * (genMax - genMin + 1)) + genMin;
    engine.addRequest(i + 1, promptLen, genLen, arrivalTime, 'SimpleQuery');
    arrivalTime += 1.0 / arrivalRate;
}

while (engine.step()) {
    if (engine.currentTime - (engine._lastLog || 0) > 1.0) {
        console.log(`t: ${engine.currentTime.toFixed(4)}, active: ${engine.activeDecodeRequests.length}, decodeWaiting: ${engine.decodeWaiting.length}, pending: ${engine.pendingIds.length}, completed: ${engine.stats.totalCompleted}`);
        engine._lastLog = engine.currentTime;
    }
}
console.log(`Finished at t: ${engine.currentTime.toFixed(4)}, completed: ${engine.stats.totalCompleted}`);
