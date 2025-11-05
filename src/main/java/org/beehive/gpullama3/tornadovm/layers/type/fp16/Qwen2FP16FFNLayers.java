package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.FP16Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.Qwen2TornadoWeights;
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
 * Qwen2FP16FFNLayers: FP16 FFN layers for Qwen2 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen3:
 * - No tempQcur/tempKcur fields in Qwen2State
 * - Includes bias terms for Q, K, V projections
 * - Standard GQA (no parallel offset RMSNorm)
 * - Uses Qwen2Kernels::processHeadsFlashAttention for attention computation
 * - Uses Qwen3Kernels::ropeRotation for position embeddings
 * - Simpler matrix dimensions (uses config.dim() and config.kvDim() directly)
 *
 * Works directly with Qwen2State to access and mutate Qwen2-specific state fields.
 */
public class Qwen2FP16FFNLayers extends AbstractLayer {

    String lastTaskGraphID;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Qwen2-specific state and config
    private final Qwen2State qwen2State;
    private final Qwen2Configuration qwen2Config;

    public Qwen2FP16FFNLayers(String taskGraphName, Qwen2State state, Qwen2TornadoWeights weights, Qwen2Configuration config) {
        super(taskGraphName, state, weights, config);

        // Store strongly-typed Qwen2 references for direct access and mutation
        this.qwen2State = state;
        this.qwen2Config = config;

        // Ensure we have Qwen2-specific weights
        if (!(weights instanceof FP16Weights weights1)) {
            throw new IllegalArgumentException(
                    "Qwen2FP16FFNLayers requires Qwen2TornadoWeights with FP16 layout");
        }

        ffnLayerTaskGraphs = setupFFNLayered();
        this.scheduler = setupGridSchedulersLayered(config);
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {

        // Single worker for tasks running with a single thread
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[1,1,1], localWorkSize=[1,1,1])
        // CUDA equivalent: kernel<<<dim3(1,1,1), dim3(1,1,1)>>>
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // config.dim / 2 Worker for RoPE
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim/2,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim/2+127)/128,1,1), dim3(128,1,1)>>>
        int h = config.numberOfHeads();
        int ic = config.headSize() / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(h, ic);
        ropeWorker.setGlobalWork(h, ic, 1);
        ropeWorker.setLocalWork(1, 1, 1);

        // config.dim Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.dim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
        configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // config.kvDim Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.kvDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.kvDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
        configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid qBiasWorker = new WorkerGrid1D(config.dim());
        qBiasWorker.setGlobalWork(config.dim(), 1, 1);
        qBiasWorker.setLocalWork(config.dim() / 8, 1, 1);
        WorkerGrid kvBiasWorker = new WorkerGrid1D(config.kvDim());
        kvBiasWorker.setGlobalWork(config.kvDim(), 1, 1);
        kvBiasWorker.setLocalWork(32, 1, 1);

        // config.hiddenDim * 32 Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.hiddenDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.hiddenDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
        configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // RMSNorm worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[256,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim+255)/256,1,1), dim3(256,1,1)>>>
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(32, 1, 1);         // Set local work size to 256 (standard efficient size)

        // Parallel attention worker configuration
        // Calculate optimal local work size based on head dimension
        int optimalLocalSize = Math.min(config.headSize(), 64); // Start with 64 threads per head
        if (config.headSize() % optimalLocalSize != 0) {
            // Find largest divisor of headSize <= 64
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

        // Copy to caches worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim+127)/128,1,1), dim3(128,1,1)>>>
        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.kvDim(), 1, 1);
        copyToCachesWorker.setLocalWork(32, 1, 1); // Set local work size to 32 (for copying to caches)

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qbias", qBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kbias", kvBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vbias", kvBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
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

        // Initialize buffers using Qwen2State directly
        qwen2State.temp.init(0.0f);
        qwen2State.tempFFN.init(0.0f);

        for (int layerIndex = 0; layerIndex < qwen2Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSingleQwen2FFNLayer((Qwen2TornadoWeights) weights, layerIndex);
            if (layerIndex == qwen2Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Qwen2 with GQA
     */
    TaskGraph setupSingleQwen2FFNLayer(Qwen2TornadoWeights weights, int layerIndex) {
     TaskGraph   unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                //Copy-in weights per layer for batched-layered layout
                weights.rms_att_weightLayered[layerIndex],
                weights.wqLayered[layerIndex],
                weights.wkLayered[layerIndex],
                weights.wvLayered[layerIndex],
                weights.woLayered[layerIndex],
                weights.q_biasLayered[layerIndex],
                weights.k_biasLayered[layerIndex],
                weights.v_biasLayered[layerIndex],
                weights.rms_ffn_weightLayered[layerIndex],
                weights.w1Layered[layerIndex],
                weights.w2Layered[layerIndex],
                weights.w3Layered[layerIndex]
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapQ, weights.wqLayered[layerIndex], config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapK, weights.wkLayered[layerIndex], config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,   state.wrapV, weights.wvLayered[layerIndex], config.dim(), config.kvDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("qbias", TransformerComputeKernelsLayered::addInPlace, state.wrapQ, weights.q_biasLayered[layerIndex], config.dim())
                .task("kbias", TransformerComputeKernelsLayered::addInPlace, state.wrapK, weights.k_biasLayered[layerIndex], config.kvDim())
                .task("vbias", TransformerComputeKernelsLayered::addInPlace, state.wrapV, weights.v_biasLayered[layerIndex], config.kvDim())
                .task("rope", Qwen3Kernels::ropeRotation,context, state.positionHolder, state.wrapQ, state.wrapK, config.numberOfKeyValueHeads(),
                        config.headSize())
                .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                        state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength())
                .task("parallel-attention", Qwen2Kernels::processHeadsFlashAttention, context,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                        state.positionHolder, layerIndex, config.contextLength())
                .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapXb,  state.wrapX, weights.woLayered[layerIndex], config.dim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                        state.wrapXb,   state.wrapHb, weights.w1Layered[layerIndex], weights.w3Layered[layerIndex], config.dim(), config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapHb, state.wrapX, weights.w2Layered[layerIndex], config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(
                        state.wrapX
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
        GridScheduler tornadoForwardScheduler = new GridScheduler();

        // Single worker for tasks running with a single thread
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[1,1,1], localWorkSize=[1,1,1])
        // CUDA equivalent: kernel<<<dim3(1,1,1), dim3(1,1,1)>>>
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // config.dim / 2 Worker for RoPE
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim/2,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim/2+127)/128,1,1), dim3(128,1,1)>>>
        int h = config.numberOfHeads();
        int ic = config.headSize() / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(h, ic);
        ropeWorker.setGlobalWork(h, ic, 1);
        ropeWorker.setLocalWork(1, 1, 1);

        // config.dim Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.dim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
        configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // config.kvDim Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.kvDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.kvDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
        configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid qBiasWorker = new WorkerGrid1D(config.dim());
        qBiasWorker.setGlobalWork(config.dim(), 1, 1);
        qBiasWorker.setLocalWork(config.dim() / 8, 1, 1);
        WorkerGrid kvBiasWorker = new WorkerGrid1D(config.kvDim());
        kvBiasWorker.setGlobalWork(config.kvDim(), 1, 1);
        kvBiasWorker.setLocalWork(32, 1, 1);

        // config.hiddenDim * 32 Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.hiddenDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.hiddenDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
        configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // RMSNorm worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[256,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim+255)/256,1,1), dim3(256,1,1)>>>
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(32, 1, 1);         // Set local work size to 256 (standard efficient size)

        // Parallel attention worker configuration
        // Calculate optimal local work size based on head dimension
        int optimalLocalSize = Math.min(config.headSize(), 64); // Start with 64 threads per head
        if (config.headSize() % optimalLocalSize != 0) {
            // Find largest divisor of headSize <= 64
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

        // Copy to caches worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim+127)/128,1,1), dim3(128,1,1)>>>
        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.kvDim(), 1, 1);
        copyToCachesWorker.setLocalWork(32, 1, 1); // Set local work size to 32 (for copying to caches)

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qbias", qBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kbias", kvBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vbias", kvBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
        }
        return tornadoForwardScheduler;
    }
}
