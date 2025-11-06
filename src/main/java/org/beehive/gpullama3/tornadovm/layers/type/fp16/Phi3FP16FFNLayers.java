package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.Phi3TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
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
 * Phi3FP16FFNLayers: FP16 FFN layers for Phi3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen2/Qwen3:
 * - Uses combined QKV matrix (wqkv) instead of separate Q, K, V matrices
 * - Includes splitQKV task to separate combined buffer
 * - Uses ropeRotationPhi3 kernel for position embeddings
 * - FFN uses single wUp matrix that outputs both Gate and Up (2 * hiddenDim)
 * - Includes splitGateUpAndSiLU task for FFN activation
 * - Uses wDown for final FFN projection
 * - No Q, K, V bias terms
 *
 * Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3FP16FFNLayers extends AbstractFFNLayers {

    String lastTaskGraphID;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Phi3-specific state and config
    private final Phi3State phi3State;
    private final Phi3Configuration phi3Config;

    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;

    public Phi3FP16FFNLayers(String taskGraphName, Phi3State state, Phi3TornadoWeights weights, Phi3Configuration config) {
        super(taskGraphName, state, weights, config);

        // Store strongly-typed Phi3 references for direct access and mutation
        this.phi3State = state;
        this.phi3Config = config;

        // Ensure we have Phi3-specific weights
        if (!(weights instanceof Phi3TornadoWeights phi3Weights)) {
            throw new IllegalArgumentException(
                    "Phi3FP16FFNLayers requires Phi3TornadoWeights with FP16 layout");
        }

        // Calculate opSize for combined QKV buffer
        // opSize = num_heads * head_dim + 2 * (num_key_value_heads * head_dim) = dim + 2 * kvDim
        this.opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());

        ffnLayerTaskGraphs = setupFFNLayered();
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

        // Combined QKV matmul worker
        int matmulQkvGlobal = opSize * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQkvRowMajorWorker = new WorkerGrid1D(matmulQkvGlobal);
        matmulQkvRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

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
        int ffnUpGlobal = (2 * config.hiddenDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnUpWorker = new WorkerGrid1D(ffnUpGlobal);
        ffnUpWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int ffnDownGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnDownWorker = new WorkerGrid1D(ffnDownGlobal);
        ffnDownWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // Map workers to tasks for each layer
        for (int i = 0; i < config.numberOfLayers(); i++) {
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".qkvmatmul", matmulQkvRowMajorWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);

            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".wGateUp", ffnUpWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".wDown", ffnDownWorker);
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

    /**
     * Setup all FFN layers for all transformer layers
     */
    List<ImmutableTaskGraph> setupFFNLayered() {
        List<ImmutableTaskGraph> ffnGraphs = new ArrayList<>();

        // Initialize buffers using Phi3State directly
        phi3State.temp.init(0.0f);
        phi3State.tempFFN.init(0.0f);

        for (int layerIndex = 0; layerIndex < phi3Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSinglePhi3FFNLayer((Phi3TornadoWeights) weights, layerIndex);
            if (layerIndex == phi3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Phi3 with combined QKV and gate/up FFN
     */
    TaskGraph setupSinglePhi3FFNLayer(Phi3TornadoWeights weights, int layerIndex) {

        TaskGraph unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(phi3State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Copy-in weights per layer for batched-layered layout
                weights.rms_att_weightLayered[layerIndex],
                weights.wqkvLayered[layerIndex],
                weights.woLayered[layerIndex],
                weights.rms_ffn_weightLayered[layerIndex],
                weights.wUpLayered[layerIndex],
                weights.wDownLayered[layerIndex]
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // RMSNorm for attention input
        unifiedLayer.task("reductionsOneBlock",
                        TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                        context,
                        phi3State.temp,
                        phi3State.wrapX,
                        phi3Config.dim(),
                        phi3Config.rmsNormEps(),
                        phi3State.localSize)
                .task("mapContext",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapX,
                        weights.rms_att_weightLayered[layerIndex],
                        phi3State.temp);

        // Combined QKV projection
        unifiedLayer.task("qkvmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapQkv,
                        weights.wqkvLayered[layerIndex],
                        phi3Config.dim(),
                        opSize,
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("splitQKV",
                        TransformerComputeKernelsLayered::splitQKV,
                        phi3State.wrapQkv,
                        phi3State.wrapQ,
                        phi3State.wrapK,
                        phi3State.wrapV,
                        phi3Config.dim(),
                        phi3Config.headSize() * phi3Config.numberOfKeyValueHeads());

        // RoPE rotation (Phi3-specific kernel)
        unifiedLayer.task("rope",
                TransformerComputeKernelsLayered::ropeRotationPhi3,
                context,
                phi3State.positionHolder,
                phi3State.wrapQ,
                phi3State.wrapK,
                phi3Config.kvDim(),
                phi3Config.headSize());

        // Copy to caches
        unifiedLayer.task("copyToCaches",
                TransformerComputeKernelsLayered::copyToCache,
                phi3State.wrapKeyCache,
                phi3State.wrapK,
                phi3State.wrapValueCache,
                phi3State.wrapV,
                phi3State.positionHolder,
                phi3Config.kvDim(),
                layerIndex,
                phi3Config.contextLength());

        // Parallel attention
        unifiedLayer.task("parallel-attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttention,
                context,
                phi3State.wrapQ,
                phi3State.wrapKeyCache,
                phi3State.wrapValueCache,
                phi3State.wrapXb,
                phi3Config.numberOfHeads(),
                phi3Config.headSize(),
                phi3Config.kvDim(),
                phi3Config.kvMul(),
                phi3State.positionHolder,
                layerIndex,
                phi3Config.contextLength());

        // Output projection
        unifiedLayer.task("matmul1",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context,
                phi3State.wrapXb,
                phi3State.wrapX,
                weights.woLayered[layerIndex],
                phi3Config.dim(),
                phi3Config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // FFN section: RMSNorm
        unifiedLayer.task("reductionsOneBlockFFN",
                        TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                        context,
                        phi3State.tempFFN,
                        phi3State.wrapX,
                        phi3Config.dim(),
                        phi3Config.rmsNormEps(),
                        phi3State.localSize)
                .task("mapContextFFN",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapX,
                        weights.rms_ffn_weightLayered[layerIndex],
                        phi3State.tempFFN);

        // FFN: combined Up and Gate projection (outputs 2 * hiddenDim)
        unifiedLayer.task("wGateUp",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapHb,
                        weights.wUpLayered[layerIndex],
                        phi3Config.dim(),
                        2 * phi3Config.hiddenDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("gateUpSiLU",
                        TransformerComputeKernelsLayered::splitGateUpAndSiLU,
                        phi3State.wrapHb,
                        phi3State.wrapHbG,
                        phi3State.wrapHbU,
                        phi3Config.hiddenDim());

        // FFN: Down projection with residual
        unifiedLayer.task("wDown",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                        context,
                        phi3State.wrapHbU,
                        phi3State.wrapX,
                        weights.wDownLayered[layerIndex],
                        phi3Config.hiddenDim(),
                        phi3Config.dim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(
                        phi3State.wrapX
                );
        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionHolder, state.temp, state.tempFFN); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    phi3State.wrapHbG, phi3State.wrapHbU, phi3State.wrapQkv); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder, // /
                    phi3State.wrapHbG, phi3State.wrapHbU, phi3State.wrapQkv);
        }
        return unifiedLayer;
    }

}
