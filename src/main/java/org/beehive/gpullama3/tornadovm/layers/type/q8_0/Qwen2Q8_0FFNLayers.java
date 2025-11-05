package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Qwen2TornadoWeightsQ8_0;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2Kernels;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layers.AbstractLayer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen2Q8_0FFNLayers: Q8_0-quantized FFN layers for Qwen2 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen2FP16FFNLayers:
 * - Uses Q8_0-quantized weights (getQuants() and getScales())
 * - Same attention and RoPE kernels as FP16 version
 * - 8-bit integer computations with dequantization
 * - 2x memory compression vs FP16
 * - Includes bias terms for Q, K, V projections
 *
 * Works directly with Qwen2State to access and mutate Qwen2-specific state fields.
 */
public class Qwen2Q8_0FFNLayers extends AbstractLayer {

    String lastTaskGraphID;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Qwen2-specific state and config
    private final Qwen2State qwen2State;
    private final Qwen2Configuration qwen2Config;

    public Qwen2Q8_0FFNLayers(String taskGraphName, Qwen2State state, Qwen2TornadoWeightsQ8_0 weights, Qwen2Configuration config) {
        super(taskGraphName, state, weights, config);

        // Store strongly-typed Qwen2 references for direct access and mutation
        this.qwen2State = state;
        this.qwen2Config = config;

        // Ensure we have Qwen2-specific quantized weights
        if (!(weights instanceof Qwen2TornadoWeightsQ8_0 qwen2WeightsQ8_0)) {
            throw new IllegalArgumentException(
                    "Qwen2Q8_0FFNLayers requires Qwen2TornadoWeightsQ8_0 with Q8_0 layout");
        }

        ffnLayerTaskGraphs = setupFFNLayered();
        this.scheduler = setupGridSchedulersLayered(config);
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        // Single worker for tasks that execute once
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // RMS norm worker
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);
        rmsNormWorker.setLocalWork(state.localSize, 1, 1);

        // Q matmul worker (standard dimensions)
        int matmulQGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = new WorkerGrid1D(matmulQGlobal);
        matmulQRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // KV matmul worker (reduced KV heads)
        int matmulKVGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = new WorkerGrid1D(matmulKVGlobal);
        matmulKVRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Bias workers
        WorkerGrid qBiasWorker = new WorkerGrid1D(config.dim());
        qBiasWorker.setGlobalWork(config.dim(), 1, 1);
        qBiasWorker.setLocalWork(config.dim() / 8, 1, 1);

        WorkerGrid kvBiasWorker = new WorkerGrid1D(config.kvDim());
        kvBiasWorker.setGlobalWork(config.kvDim(), 1, 1);
        kvBiasWorker.setLocalWork(32, 1, 1);

        // RoPE worker (2D: heads x embedding_head/2)
        int ic = config.headSize() / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(config.numberOfHeads(), ic);
        ropeWorker.setGlobalWork(config.numberOfHeads(), ic, 1);
        ropeWorker.setLocalWork(8, 1, 1);

        // Copy to cache worker
        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.kvDim(), 1, 1);
        copyToCachesWorker.setLocalWork(32, 1, 1);

        // Parallel attention worker
        int optimalLocalSize = Math.min(config.headSize(), 64);
        if (config.headSize() % optimalLocalSize != 0) {
            for (int size = 64; size >= 1; size--) {
                if (config.headSize() % size == 0) {
                    optimalLocalSize = size;
                    break;
                }
            }
        }
        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * optimalLocalSize, 1, 1);
        parallelAttentionWorker.setLocalWork(optimalLocalSize, 1, 1);

        // Matmul1 worker (output projection)
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = new WorkerGrid1D(matmul1Global);
        matmul1Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // FFN workers
        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker = new WorkerGrid1D(fusedFFNW1W3Global);
        fusedFFNW1W3Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker = new WorkerGrid1D(projectionTwoGlobal);
        projectionTwoWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Map workers to tasks for each layer
        for (int i = 0; i < config.numberOfLayers(); i++) {
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".qmatmul", matmulQRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".kmatmul", matmulKVRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".vmatmul", matmulKVRowMajorWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".qbias", qBiasWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".kbias", kvBiasWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".vbias", kvBiasWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);

            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", fusedFFNW1W3Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", projectionTwoWorker);
        }

        return gridScheduler;
    }

    @Override
    public GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return ffnLayerTaskGraph;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return null;
    }

    public List<ImmutableTaskGraph> getFfnLayerTaskGraphs() {
        return ffnLayerTaskGraphs;
    }

    public String getLastTaskGraphID() {
        return lastTaskGraphID;
    }

    private void setupLastID(String taskGraphID) {
        if (lastTaskGraphID == null) {
            lastTaskGraphID = taskGraphID;
        } else {
            if (!lastTaskGraphID.equals(taskGraphID)) {
                throw new IllegalStateException("Task graph IDs do not match: " + lastTaskGraphID + " vs " + taskGraphID);
            }
        }
    }

    /**
     * Setup all FFN layers for all transformer layers
     */
    List<ImmutableTaskGraph> setupFFNLayered() {
        List<ImmutableTaskGraph> ffnGraphs = new ArrayList<>();

        // Initialize buffers using Qwen2State directly
        qwen2State.temp.init(0.0f);
        qwen2State.tempFFN.init(0.0f);

        for (int layerIndex = 0; layerIndex < qwen2Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSingleQwen2Q8_0FFNLayer((Qwen2TornadoWeightsQ8_0) weights, layerIndex);
            if (layerIndex == qwen2Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Qwen2 with Q8_0 quantization and GQA
     */
    TaskGraph setupSingleQwen2Q8_0FFNLayer(Qwen2TornadoWeightsQ8_0 weights, int layerIndex) {

        TaskGraph unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                //Copy-in weights per layer for batched-layered layout (quantized + scales)
                weights.rms_att_weightLayered[layerIndex],
                weights.wqLayered[layerIndex].getQuants(),
                weights.wqLayered[layerIndex].getScales(),
                weights.wkLayered[layerIndex].getQuants(),
                weights.wkLayered[layerIndex].getScales(),
                weights.wvLayered[layerIndex].getQuants(),
                weights.wvLayered[layerIndex].getScales(),
                weights.woLayered[layerIndex].getQuants(),
                weights.woLayered[layerIndex].getScales(),
                weights.q_biasLayered[layerIndex],
                weights.k_biasLayered[layerIndex],
                weights.v_biasLayered[layerIndex],
                weights.rms_ffn_weightLayered[layerIndex],
                weights.w1Layered[layerIndex].getQuants(),
                weights.w1Layered[layerIndex].getScales(),
                weights.w2Layered[layerIndex].getQuants(),
                weights.w2Layered[layerIndex].getScales(),
                weights.w3Layered[layerIndex].getQuants(),
                weights.w3Layered[layerIndex].getScales()
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // RMSNorm for attention input
        unifiedLayer.task("reductionsOneBlock",
                        TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                        context,
                        qwen2State.temp,
                        qwen2State.wrapX,
                        qwen2Config.dim(),
                        qwen2Config.rmsNormEps(),
                        qwen2State.localSize)
                .task("mapContext",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context,
                        qwen2State.wrapXb,
                        qwen2State.wrapX,
                        weights.rms_att_weightLayered[layerIndex],
                        qwen2State.temp);

        // Q, K, V projections (quantized)
        unifiedLayer.task("qmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        qwen2State.wrapXb,
                        qwen2State.wrapQ,
                        weights.wqLayered[layerIndex].getQuants(),
                        weights.wqLayered[layerIndex].getScales(),
                        qwen2Config.dim(),
                        qwen2Config.dim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        qwen2State.wrapXb,
                        qwen2State.wrapK,
                        weights.wkLayered[layerIndex].getQuants(),
                        weights.wkLayered[layerIndex].getScales(),
                        qwen2Config.dim(),
                        qwen2Config.kvDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        qwen2State.wrapXb,
                        qwen2State.wrapV,
                        weights.wvLayered[layerIndex].getQuants(),
                        weights.wvLayered[layerIndex].getScales(),
                        qwen2Config.dim(),
                        qwen2Config.kvDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Bias terms for Q, K, V
        unifiedLayer.task("qbias",
                        TransformerComputeKernelsLayered::addInPlace,
                        qwen2State.wrapQ,
                        weights.q_biasLayered[layerIndex],
                        qwen2Config.dim())
                .task("kbias",
                        TransformerComputeKernelsLayered::addInPlace,
                        qwen2State.wrapK,
                        weights.k_biasLayered[layerIndex],
                        qwen2Config.kvDim())
                .task("vbias",
                        TransformerComputeKernelsLayered::addInPlace,
                        qwen2State.wrapV,
                        weights.v_biasLayered[layerIndex],
                        qwen2Config.kvDim());

        // RoPE rotation task graph
        unifiedLayer.task("rope",
                Qwen3Kernels::ropeRotation,
                context,
                qwen2State.positionHolder,
                qwen2State.wrapQ,
                qwen2State.wrapK,
                qwen2Config.numberOfKeyValueHeads(),
                qwen2Config.headSize());

        // Copy to caches
        unifiedLayer.task("copyToCaches",
                TransformerComputeKernelsLayered::copyToCache,
                qwen2State.wrapKeyCache,
                qwen2State.wrapK,
                qwen2State.wrapValueCache,
                qwen2State.wrapV,
                qwen2State.positionHolder,
                qwen2Config.kvDim(),
                layerIndex,
                qwen2Config.contextLength());

        // Parallel attention using Qwen2 kernel
        unifiedLayer.task("parallel-attention",
                Qwen2Kernels::processHeadsFlashAttention,
                context,
                qwen2State.wrapQ,
                qwen2State.wrapKeyCache,
                qwen2State.wrapValueCache,
                qwen2State.wrapXb,
                qwen2Config.numberOfHeads(),
                qwen2Config.headSize(),
                qwen2Config.kvDim(),
                qwen2Config.kvMul(),
                qwen2State.positionHolder,
                layerIndex,
                qwen2Config.contextLength());

        // Output projection (quantized)
        unifiedLayer.task("matmul1",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context,
                qwen2State.wrapXb,
                qwen2State.wrapX,
                weights.woLayered[layerIndex].getQuants(),
                weights.woLayered[layerIndex].getScales(),
                qwen2Config.dim(),
                qwen2Config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // FFN section: RMSNorm
        unifiedLayer.task("reductionsOneBlockFFN",
                        TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                        context,
                        qwen2State.tempFFN,
                        qwen2State.wrapX,
                        qwen2Config.dim(),
                        qwen2Config.rmsNormEps(),
                        qwen2State.localSize)
                .task("reductionFinalNormalizationFFN",
                        TransformerComputeKernelsLayered::reductionFinalNormalization,
                        context,
                        qwen2State.tempFFN,
                        qwen2Config.dim(),
                        qwen2Config.rmsNormEps())
                .task("mapContextFFN",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context,
                        qwen2State.wrapXb,
                        qwen2State.wrapX,
                        weights.rms_ffn_weightLayered[layerIndex],
                        qwen2State.tempFFN);

        // Fused FFN with GLU activation (quantized)
        unifiedLayer.task("fused_ffn_w1_w3",
                        TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation,
                        context,
                        qwen2State.wrapXb,
                        qwen2State.wrapHb,
                        weights.w1Layered[layerIndex].getQuants(),
                        weights.w1Layered[layerIndex].getScales(),
                        weights.w3Layered[layerIndex].getQuants(),
                        weights.w3Layered[layerIndex].getScales(),
                        qwen2Config.dim(),
                        qwen2Config.hiddenDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                        context,
                        qwen2State.wrapHb,
                        qwen2State.wrapX,
                        weights.w2Layered[layerIndex].getQuants(),
                        weights.w2Layered[layerIndex].getScales(),
                        qwen2Config.hiddenDim(),
                        qwen2Config.dim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(
                        qwen2State.wrapX
                );
        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    qwen2State.positionHolder, qwen2State.temp, qwen2State.tempFFN);

            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context, qwen2State.wrapXb,
                    qwen2State.wrapQ, qwen2State.wrapK, qwen2State.wrapV,
                    qwen2State.wrapKeyCache, qwen2State.wrapValueCache,
                    qwen2State.wrapAtt, qwen2State.wrapHb);
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(context, qwen2State.wrapXb,
                    qwen2State.wrapQ, qwen2State.wrapK, qwen2State.wrapV,
                    qwen2State.wrapKeyCache, qwen2State.wrapValueCache,
                    qwen2State.wrapAtt, qwen2State.wrapHb, qwen2State.positionHolder);
        }
        return unifiedLayer;
    }

    /**
     * Setup GridScheduler with Qwen2-specific worker configurations
     */
    private GridScheduler setupGridSchedulersLayered(Qwen2Configuration config) {
        GridScheduler gridScheduler = new GridScheduler();

        // Single worker for tasks that execute once
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // RMS norm worker
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);
        rmsNormWorker.setLocalWork(state.localSize, 1, 1);

        // Q matmul worker (standard dimensions)
        int matmulQGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = new WorkerGrid1D(matmulQGlobal);
        matmulQRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // KV matmul worker (reduced KV heads)
        int matmulKVGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = new WorkerGrid1D(matmulKVGlobal);
        matmulKVRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Bias workers
        WorkerGrid qBiasWorker = new WorkerGrid1D(config.dim());
        qBiasWorker.setGlobalWork(config.dim(), 1, 1);
        qBiasWorker.setLocalWork(config.dim() / 8, 1, 1);

        WorkerGrid kvBiasWorker = new WorkerGrid1D(config.kvDim());
        kvBiasWorker.setGlobalWork(config.kvDim(), 1, 1);
        kvBiasWorker.setLocalWork(32, 1, 1);

        // RoPE worker (2D: heads x embedding_head/2)
        int ic = config.headSize() / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(config.numberOfHeads(), ic);
        ropeWorker.setGlobalWork(config.numberOfHeads(), ic, 1);
        ropeWorker.setLocalWork(8, 1, 1);

        // Copy to cache worker
        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.kvDim(), 1, 1);
        copyToCachesWorker.setLocalWork(32, 1, 1);

        // Parallel attention worker
        int optimalLocalSize = Math.min(config.headSize(), 64);
        if (config.headSize() % optimalLocalSize != 0) {
            for (int size = 64; size >= 1; size--) {
                if (config.headSize() % size == 0) {
                    optimalLocalSize = size;
                    break;
                }
            }
        }
        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * optimalLocalSize, 1, 1);
        parallelAttentionWorker.setLocalWork(optimalLocalSize, 1, 1);

        // Matmul1 worker (output projection)
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = new WorkerGrid1D(matmul1Global);
        matmul1Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // FFN workers
        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker = new WorkerGrid1D(fusedFFNW1W3Global);
        fusedFFNW1W3Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker = new WorkerGrid1D(projectionTwoGlobal);
        projectionTwoWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Map workers to tasks for each layer
        for (int i = 0; i < config.numberOfLayers(); i++) {
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".qmatmul", matmulQRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".kmatmul", matmulKVRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".vmatmul", matmulKVRowMajorWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".qbias", qBiasWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".kbias", kvBiasWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".vbias", kvBiasWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);

            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", fusedFFNW1W3Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", projectionTwoWorker);
        }

        return gridScheduler;
    }
}
