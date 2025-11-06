package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Phi3TornadoWeightsQ8_0;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
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
 * Phi3Q8_0FFNLayers: Q8_0-quantized FFN layers for Phi3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Phi3FP16FFNLayers:
 * - Uses Q8_0-quantized weights (getQuants() and getScales())
 * - Same attention and RoPE kernels as FP16 version
 * - 8-bit integer computations with dequantization
 * - 2x memory compression vs FP16
 * - Same combined QKV and gate/up FFN structure
 *
 * Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3Q8_0FFNLayers extends AbstractLayer {

    String lastTaskGraphID;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Phi3-specific state and config
    private final Phi3State phi3State;
    private final Phi3Configuration phi3Config;

    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;

    public Phi3Q8_0FFNLayers(String taskGraphName, Phi3State state, Phi3TornadoWeightsQ8_0 weights, Phi3Configuration config) {
        super(taskGraphName, state, weights, config);

        // Store strongly-typed Phi3 references for direct access and mutation
        this.phi3State = state;
        this.phi3Config = config;

//        // Ensure we have Phi3-specific quantized weights
//        if (!(weights instanceof Phi3TornadoWeightsQ8_0 phi3WeightsQ8_0)) {
//            throw new IllegalArgumentException(
//                    "Phi3Q8_0FFNLayers requires Phi3TornadoWeightsQ8_0 with Q8_0 layout");
//        }

//        var phi3Weights = requireWeightsType(weights, Phi3TornadoWeightsQ8_0.class, "Phi3Q8_0FFNLayers", "Q8_0");


        // Calculate opSize for combined QKV buffer
        // opSize = num_heads * head_dim + 2 * (num_key_value_heads * head_dim) = dim + 2 * kvDim
        this.opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());

        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        // CUDA equivalent: kernel<<<dim3(1,1,1), dim3(1,1,1)>>>
        WorkerGrid singleWorker = new WorkerGrid1D(1);
        singleWorker.setGlobalWork(1, 1, 1);
        singleWorker.setLocalWork(1, 1, 1);

        // config.dim / 2 Worker for RoPE
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim/2,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim/2+127)/128,1,1), dim3(128,1,1)>>>
        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim() / 2);
        ropeWorker.setGlobalWork(config.dim() / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

        // config.dim Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.dim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
        configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        final int opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());

        int qkvmatmulDimRowMajorGlobal = opSize * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid qkvDimRowMajorGlobalWorker = new WorkerGrid1D(qkvmatmulDimRowMajorGlobal);
        qkvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // config.kvDim Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.kvDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.kvDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
        configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // config.hiddenDim * 32 Worker for Row major access
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.hiddenDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
        // CUDA equivalent: kernel<<<dim3(config.hiddenDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
        configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int wgetUPDimRowMajor = 2 * config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid wgetHiddenDimRowMajorWorker = new WorkerGrid1D(wgetUPDimRowMajor);
        wgetHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // RMSNorm worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[256,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim+255)/256,1,1), dim3(256,1,1)>>>
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);  // Set global work size to total dimension
        rmsNormWorker.setLocalWork(256, 1, 1);         // Set local work size to 256 (standard efficient size)

        // Parallel attention worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.numberOfHeads,1,1], localWorkSize=[4,1,1])
        // CUDA equivalent: kernel<<<dim3((config.numberOfHeads+3)/4,1,1), dim3(4,1,1)>>>
        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        // the global group work size is numberOfHeads * localWorkGroupSize, where the localWorkGroupSize is currently 4
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * 8, 1, 1);
        parallelAttentionWorker.setLocalWork(8, 1, 1); // Set local work size to 4 (for parallel attention)

        // Copy to caches worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim+127)/128,1,1), dim3(128,1,1)>>>
        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.dim(), 1, 1);
        copyToCachesWorker.setLocalWork(128, 1, 1); // Set local work size to 32 (for copying to caches)

        // Q copy worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((config.dim+127)/128,1,1), dim3(128,1,1)>>>
        WorkerGrid copyQWorker = new WorkerGrid1D(config.dim());
        copyQWorker.setGlobalWork(config.dim(), 1, 1);
        copyQWorker.setLocalWork(128, 1, 1);

        // K copy worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[kvSize,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((kvSize+127)/128,1,1), dim3(128,1,1)>>>
        int kvSize = config.headSize() * config.numberOfKeyValueHeads();
        WorkerGrid copyKWorker = new WorkerGrid1D(kvSize);
        copyKWorker.setGlobalWork(kvSize, 1, 1);
        copyKWorker.setLocalWork(128, 1, 1);

        // V copy worker configuration
        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[kvSize,1,1], localWorkSize=[128,1,1])
        // CUDA equivalent: kernel<<<dim3((kvSize+127)/128,1,1), dim3(128,1,1)>>>
        WorkerGrid copyVWorker = new WorkerGrid1D(kvSize);
        copyVWorker.setGlobalWork(kvSize, 1, 1);
        copyVWorker.setLocalWork(128, 1, 1);

        WorkerGrid hiddenDimWorker = new WorkerGrid1D(config.hiddenDim());
        hiddenDimWorker.setGlobalWork(config.hiddenDim(), 1, 1);
        hiddenDimWorker.setLocalWork(128, 1, 1);

        WorkerGrid splitGateUpSiLUWorker = new WorkerGrid1D(config.hiddenDim());
        splitGateUpSiLUWorker.setGlobalWork(config.hiddenDim(), 1, 1);
        splitGateUpSiLUWorker.setLocalWork(128, 1, 1);

        // Total work size is dimQ + 2*dimKV (same as opSize)
        WorkerGrid splitQKVWorker = new WorkerGrid1D(opSize);
        splitQKVWorker.setGlobalWork(opSize, 1, 1);
        splitQKVWorker.setLocalWork(128, 1, 1);

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkvmatmul", qkvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".splitQKV", splitQKVWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".wDown", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".wGateUp", wgetHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            // New FFN tasks
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".gateUpSiLU", splitGateUpSiLUWorker);
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

    /**
     * Setup all FFN layers for all transformer layers
     */
    List<ImmutableTaskGraph> setupFFNLayered() {
        List<ImmutableTaskGraph> ffnGraphs = new ArrayList<>();

        // Initialize buffers using Phi3State directly
        phi3State.temp.init(0.0f);
        phi3State.tempFFN.init(0.0f);

        for (int layerIndex = 0; layerIndex < phi3Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSinglePhi3Q8_0FFNLayer((Phi3TornadoWeightsQ8_0) weights, layerIndex);
            if (layerIndex == phi3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Phi3 with Q8_0 quantization, combined QKV and gate/up FFN
     */
    TaskGraph setupSinglePhi3Q8_0FFNLayer(Phi3TornadoWeightsQ8_0 weights, int layerIndex) {

        TaskGraph unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(phi3State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Copy-in quantized weights per layer
                weights.rms_att_weightLayered[layerIndex],
                weights.wqkvLayered[layerIndex].getQuants(),
                weights.wqkvLayered[layerIndex].getScales(),
                weights.woLayered[layerIndex].getQuants(),
                weights.woLayered[layerIndex].getScales(),
                weights.rms_ffn_weightLayered[layerIndex],
                weights.wUpLayered[layerIndex].getQuants(),
                weights.wUpLayered[layerIndex].getScales(),
                weights.wDownLayered[layerIndex].getQuants(),
                weights.wDownLayered[layerIndex].getScales()
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

        // Combined QKV projection (quantized)
        unifiedLayer.task("qkvmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapQkv,
                        weights.wqkvLayered[layerIndex].getQuants(),
                        weights.wqkvLayered[layerIndex].getScales(),
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

        // Output projection (quantized)
        unifiedLayer.task("matmul1",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context,
                phi3State.wrapXb,
                phi3State.wrapX,
                weights.woLayered[layerIndex].getQuants(),
                weights.woLayered[layerIndex].getScales(),
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

        // FFN: combined Up and Gate projection (outputs 2 * hiddenDim, quantized)
        unifiedLayer.task("wGateUp",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapHb,
                        weights.wUpLayered[layerIndex].getQuants(),
                        weights.wUpLayered[layerIndex].getScales(),
                        phi3Config.dim(),
                        2 * phi3Config.hiddenDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("gateUpSiLU",
                        TransformerComputeKernelsLayered::splitGateUpAndSiLU,
                        phi3State.wrapHb,
                        phi3State.wrapHbG,
                        phi3State.wrapHbU,
                        phi3Config.hiddenDim());

        // FFN: Down projection with residual (quantized)
        unifiedLayer.task("wDown",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                        context,
                        phi3State.wrapHbU,
                        phi3State.wrapX,
                        weights.wDownLayered[layerIndex].getQuants(),
                        weights.wDownLayered[layerIndex].getScales(),
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
