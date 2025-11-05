package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Qwen3Q8_0TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
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
 * Qwen3Q8_0FFNLayers: Q8_0-quantized FFN layers for Qwen3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen3FP16FFNLayers:
 * - Uses Q8_0-quantized weights (getQuants() and getScales())
 * - Same Qwen3Kernels for RMSNorm and RoPE
 * - 8-bit integer computations with dequantization
 * - 2x memory compression vs FP16
 *
 * Works directly with Qwen3State to access and mutate Qwen3-specific state fields
 * like tempQcur and tempKcur.
 */
public class Qwen3Q8_0FFNLayers extends AbstractLayer {

    String lastTaskGraphID;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Qwen3-specific state and config
    private final Qwen3State qwen3State;
    private final Qwen3Configuration qwen3Config;

    // Qwen3-specific GQA parameters
    private final int nHeadKv;
    private final int nEmbdHeadK;
    private final int nEmbdHeadV;
    private final int nEmbdVGqa;
    private final int nEmbdHead;
    private final int nEmbdGqa;
    private final int gqa;

    public Qwen3Q8_0FFNLayers(String taskGraphName, Qwen3State state, Qwen3Q8_0TornadoWeights weights, Qwen3Configuration config) {
        super(taskGraphName, state, weights, config);

        // Store strongly-typed Qwen3 references for direct access and mutation
        this.qwen3State = state;
        this.qwen3Config = config;

        // Initialize GQA parameters
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue();
        this.nEmbdVGqa = nEmbdHeadV * nHeadKv;
        this.nEmbdHead = nEmbdHeadV;
        this.nEmbdGqa = nEmbdVGqa;
        this.gqa = config.numberOfHeads() / config.numberOfKeyValueHeads();

        ffnLayerTaskGraphs = setupFFNLayered();
        this.scheduler = setupGridSchedulersLayered(config);
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {

        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(state.localSize, 1, 1);         // Set local work size to 256 (standard efficient size)

        int matmulQGlobal = nEmbdHeadK * config.numberOfHeads() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = new WorkerGrid1D(matmulQGlobal);
        matmulQRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int matmulKVGlobal = nEmbdGqa * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = new WorkerGrid1D(matmulKVGlobal);
        matmulKVRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid curWorker = new WorkerGrid1D(nEmbdHead);  // mEmbdHead = 128
        curWorker.setGlobalWork(nEmbdHead, 1, 1);  // Set global work size to total dimension
        curWorker.setLocalWork(128, 1, 1);         // Set local work size to 256 (standard efficient size)

        // Qcur
        WorkerGrid qCurWorker = new WorkerGrid1D(config.numberOfHeads() * nEmbdHead);
        qCurWorker.setLocalWork(nEmbdHead, 1, 1);

        // Kcur
        WorkerGrid kCurWorker = new WorkerGrid1D(config.numberOfKeyValueHeads() * nEmbdHead);
        kCurWorker.setLocalWork(nEmbdHead, 1, 1);

        int h = config.numberOfHeads();
        int ic = nEmbdHead / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(h, ic);
        ropeWorker.setGlobalWork(h, ic, 1);
        ropeWorker.setLocalWork(8, 1, 1);

        WorkerGrid copyToCachesWorker = new WorkerGrid1D(nEmbdGqa);
        copyToCachesWorker.setGlobalWork(nEmbdGqa, 1, 1);
        copyToCachesWorker.setLocalWork(128, 1, 1);

        // Parallel attention worker configuration
        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * 32, 1, 1);
        parallelAttentionWorker.setLocalWork(32, 1, 1);

        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = new WorkerGrid1D(matmul1Global);
        matmul1Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker = new WorkerGrid1D(fusedFFNW1W3Global);
        fusedFFNW1W3Worker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker = new WorkerGrid1D(projectionTwoGlobal);
        projectionTwoWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);

            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", matmulQRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", matmulKVRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", matmulKVRowMajorWorker);

            // Qcur
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Qcur", qCurWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Qcur", qCurWorker);

            // Kcur
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Kcur", kCurWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Kcur", kCurWorker);

            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ropeRotation", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", fusedFFNW1W3Worker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", projectionTwoWorker);
        }
        return tornadoForwardScheduler;
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

        // Initialize buffers using Qwen3State directly
        qwen3State.temp.init(0.0f);
        qwen3State.tempFFN.init(0.0f);
        qwen3State.tempQcur.init(0.0f);
        qwen3State.tempKcur.init(0.0f);

        for (int layerIndex = 0; layerIndex < qwen3Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSingleQwen3FFNLayer((Qwen3Q8_0TornadoWeights) weights, layerIndex);
            if (layerIndex == qwen3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Qwen3 with GQA (Q8_0 quantized)
     */
    TaskGraph setupSingleQwen3FFNLayer(Qwen3Q8_0TornadoWeights weights, int layerIndex) {

        TaskGraph unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(qwen3State.wrapX);

        // Transfer Q8_0 weights for this layer (quants and scales)
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex],
                weights.wqLayered[layerIndex].getQuants(),
                weights.wqLayered[layerIndex].getScales(),
                weights.wkLayered[layerIndex].getQuants(),
                weights.wkLayered[layerIndex].getScales(),
                weights.wvLayered[layerIndex].getQuants(),
                weights.wvLayered[layerIndex].getScales(),
                weights.woLayered[layerIndex].getQuants(),
                weights.woLayered[layerIndex].getScales(),
                weights.rms_att_KNormLayered[layerIndex],
                weights.rms_att_QNormLayered[layerIndex],
                weights.rms_ffn_weightLayered[layerIndex],
                weights.w1Layered[layerIndex].getQuants(),
                weights.w1Layered[layerIndex].getScales(),
                weights.w2Layered[layerIndex].getQuants(),
                weights.w2Layered[layerIndex].getScales(),
                weights.w3Layered[layerIndex].getQuants(),
                weights.w3Layered[layerIndex].getScales());

        // Configure layer data transfers (EVERY_EXECUTION and device persistence)
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ========== ATTENTION BLOCK ==========

        // RMS norm for attention input
        unifiedLayer.task("reductionsOneBlock",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, state.temp, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContext",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context, state.wrapXb, state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp);

        // QKV projections with Qwen3 GQA dimensions
        // Q8_0 weights pass both quants and scales
        int qDim0 = nEmbdHeadK * config.numberOfHeads();  // Query dimension
        int kvDim0 = nEmbdGqa;                             // KV dimension (smaller due to GQA)
        int qkvDim1 = config.dim();                        // Input dimension

        unifiedLayer.task("qmatmul",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context, state.wrapXb, state.wrapQ,
                weights.wqLayered[layerIndex].getQuants(), weights.wqLayered[layerIndex].getScales(),
                qkvDim1, qDim0, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context, state.wrapXb, state.wrapK,
                        weights.wkLayered[layerIndex].getQuants(), weights.wkLayered[layerIndex].getScales(),
                        qkvDim1, kvDim0, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context, state.wrapXb, state.wrapV,
                        weights.wvLayered[layerIndex].getQuants(), weights.wvLayered[layerIndex].getScales(),
                        qkvDim1, kvDim0, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Qcur: RMS norm with parallel offset for Query
        Qwen3State qwen3State = (Qwen3State) state;
        unifiedLayer.task("rmsnormReduction_Qcur",
                Qwen3Kernels::rmsnormWithParallelOffset,
                context, qwen3State.tempQcur, state.wrapQ, state.localSize, nEmbdHead, config.rmsNormEps())
                .task("rmsnormMapIndexInPlace_Qcur",
                        Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset,
                        context, state.wrapQ, weights.rms_att_QNormLayered[layerIndex], nEmbdHead, qwen3State.tempQcur);

        // Kcur: RMS norm with parallel offset for Key
        unifiedLayer.task("rmsnormReduction_Kcur",
                Qwen3Kernels::rmsnormWithParallelOffset,
                context, qwen3State.tempKcur, state.wrapK, state.localSize, nEmbdHead, config.rmsNormEps())
                .task("rmsnormMapIndexInPlace_Kcur",
                        Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset,
                        context, state.wrapK, weights.rms_att_KNormLayered[layerIndex], nEmbdHead, qwen3State.tempKcur);

        // RoPE rotation (Qwen3 variant)
        unifiedLayer.task("ropeRotation",
                Qwen3Kernels::ropeRotation,
                context, state.positionHolder, state.wrapQ, state.wrapK,
                config.numberOfKeyValueHeads(), nEmbdHead);

        // Copy to KV cache
        unifiedLayer.task("copyToCaches",
                TransformerComputeKernelsLayered::copyToCache,
                state.wrapKeyCache, state.wrapK, state.wrapValueCache, state.wrapV,
                state.positionHolder, nEmbdGqa, layerIndex, config.contextLength());

        // Parallel attention (with GQA support)
        unifiedLayer.task("parallel-attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttentionOpt,
                context, state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                config.numberOfHeads(), nEmbdHead, nEmbdGqa, gqa, state.positionHolder, layerIndex, config.contextLength());

        // Output projection (Q8_0 weights)
        unifiedLayer.task("matmul1",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context, state.wrapXb, state.wrapX,
                weights.woLayered[layerIndex].getQuants(), weights.woLayered[layerIndex].getScales(),
                qDim0, config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        // ========== FEED-FORWARD BLOCK ==========

        // RMS norm for FFN input
        unifiedLayer.task("reductionsOneBlockFFN",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, state.tempFFN, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextFFN",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context, state.wrapXb, state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN);

        // Fused FFN: w1(x) ⊗ w3(x) with SiLU activation (Q8_0 weights)
        unifiedLayer.task("fused_ffn_w1_w3",
                TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation,
                context, state.wrapXb, state.wrapHb,
                weights.w1Layered[layerIndex].getQuants(), weights.w1Layered[layerIndex].getScales(),
                weights.w3Layered[layerIndex].getQuants(), weights.w3Layered[layerIndex].getScales(),
                config.dim(), config.hiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                        context, state.wrapHb, state.wrapX,
                        weights.w2Layered[layerIndex].getQuants(), weights.w2Layered[layerIndex].getScales(),
                        config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(state.wrapX);

        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN);

            Qwen3State qwen3State = (Qwen3State) state;
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    qwen3State.tempQcur, qwen3State.tempKcur);

            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context, state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb);
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb, state.positionHolder);

            Qwen3State qwen3State = (Qwen3State) state;
            unifiedLayer.consumeFromDevice(qwen3State.tempQcur, qwen3State.tempKcur);
        }
        return unifiedLayer;
    }

    /**
     * Setup GridScheduler with Qwen3-specific worker configurations
     */
    private GridScheduler setupGridSchedulersLayered(Qwen3Configuration config) {
        GridScheduler gridScheduler = new GridScheduler();

        // Single worker for tasks that execute once
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // RMS norm worker
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);
        rmsNormWorker.setLocalWork(qwen3State.localSize, 1, 1);

        // Q matmul worker (GQA: full query heads)
        int matmulQGlobal = nEmbdHeadK * config.numberOfHeads() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = new WorkerGrid1D(matmulQGlobal);
        matmulQRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // KV matmul worker (GQA: reduced KV heads)
        int matmulKVGlobal = nEmbdGqa * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = new WorkerGrid1D(matmulKVGlobal);
        matmulKVRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Current embedding head worker
        WorkerGrid curWorker = new WorkerGrid1D(nEmbdHead);
        curWorker.setGlobalWork(nEmbdHead, 1, 1);
        curWorker.setLocalWork(128, 1, 1);

        // Q current worker
        WorkerGrid qCurWorker = new WorkerGrid1D(config.numberOfHeads() * nEmbdHead);
        qCurWorker.setLocalWork(nEmbdHead, 1, 1);

        // K current worker
        WorkerGrid kCurWorker = new WorkerGrid1D(config.numberOfKeyValueHeads() * nEmbdHead);
        kCurWorker.setLocalWork(nEmbdHead, 1, 1);

        // RoPE worker (2D: heads x embedding_head/2)
        int ic = nEmbdHead / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(config.numberOfHeads(), ic);
        ropeWorker.setGlobalWork(config.numberOfHeads(), ic, 1);
        ropeWorker.setLocalWork(8, 1, 1);

        // Copy to cache worker
        WorkerGrid copyToCachesWorker = new WorkerGrid1D(nEmbdGqa);
        copyToCachesWorker.setGlobalWork(nEmbdGqa, 1, 1);
        copyToCachesWorker.setLocalWork(128, 1, 1);

        // Parallel attention worker
        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * 32, 1, 1);
        parallelAttentionWorker.setLocalWork(32, 1, 1);

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

            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Qcur", qCurWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Qcur", qCurWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Kcur", kCurWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Kcur", kCurWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".ropeRotation", ropeWorker);
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