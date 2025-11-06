package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.FP16Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layers.AbstractLayer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

public class LlamaFP16FFNLayers extends AbstractLayer {

    String lastTaskGraphID;
    TaskGraph ffnTaskGraphs;
    GridScheduler scheduler;
   List<ImmutableTaskGraph> ffnLayerTaskGraphs;
    public LlamaFP16FFNLayers(String taskGraph, State state, Weights weights, Configuration config) {
        super(taskGraph, state, weights, config);

        // Ensure we have the Tornado-specific weights layout
        if (!(weights instanceof FP16Weights llamaWeights)) {
            throw new IllegalArgumentException(
                    "LlamaFFNLayer requires LlamaTornadoWeights with layered layout");
        }

        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {

        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(config.dim()/2, 128);
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configKvDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = WorkerGridFactory.genericWorker(configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

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

        // Map workers to tasks
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
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
    public  GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public  TaskGraph getTaskGraph() {
        return ffnTaskGraphs;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return null;
    }

    public List<ImmutableTaskGraph> getFfnLayerTaskGraphs() {
        return ffnLayerTaskGraphs;
    }

    List<ImmutableTaskGraph> setupFFNLayered() {
        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        var numLayers = config.numberOfLayers();

        return IntStream.range(0, numLayers)
            .mapToObj(i -> {
                var ffnLayer = setupSingleFFNLayer((LlamaTornadoWeights) weights, config, i);
                if (i == numLayers - 1) setupLastID(ffnLayer.getTaskGraphName());
                return ffnLayer.snapshot();
            })
            .toList();
    }

    public String getLastTaskGraphID() {
            return lastTaskGraphID;
    }

    private void setupLastID(String taskGraphID) {
        if (lastTaskGraphID == null) lastTaskGraphID = taskGraphID;
        else if (!lastTaskGraphID.equals(taskGraphID))
            throw new IllegalStateException("Task graph IDs do not match: " + lastTaskGraphID + " vs " + taskGraphID);
    }

    TaskGraph setupSingleFFNLayer(FP16Weights weights, Configuration config, int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex],
                weights.wqLayered[layerIndex],
                weights.wkLayered[layerIndex],
                weights.wvLayered[layerIndex],
                weights.woLayered[layerIndex],
                weights.rms_ffn_weightLayered[layerIndex],
                weights.w1Layered[layerIndex],
                weights.w2Layered[layerIndex],
                weights.w3Layered[layerIndex]);
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
        unifiedLayer
                .task("reductionsOneBlock", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb, state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, state.wrapXb, state.wrapQ, weights.wqLayered[layerIndex], config.dim(), config.dim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, state.wrapXb, state.wrapK, weights.wkLayered[layerIndex], config.dim(), config.kvDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, state.wrapXb, state.wrapV, weights.wvLayered[layerIndex], config.dim(), config.kvDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("rope", TransformerComputeKernelsLayered::ropeRotation, context, state.positionHolder, state.wrapQ, state.wrapK, config.kvDim(), config.headSize())
                .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache, state.wrapKeyCache, state.wrapK, state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(),
                        layerIndex, config.contextLength())
                .task("parallel-attention", TransformerComputeKernelsLayered::processHeadsFlashAttention, context, state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(), state.positionHolder, layerIndex, config.contextLength())
                .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, state.wrapXb, state.wrapX, weights.woLayered[layerIndex], config.dim(), config.dim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN, state.wrapX, config.dim(), config.rmsNormEps(),
                        state.localSize)
                .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb, state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context, state.wrapXb, state.wrapHb, weights.w1Layered[layerIndex],
                        weights.w3Layered[layerIndex], config.dim(), config.hiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, state.wrapHb, state.wrapX, weights.w2Layered[layerIndex], config.hiddenDim(),
                        config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC).persistOnDevice(state.wrapX);
        return unifiedLayer;
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionHolder, state.temp, state.tempFFN); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder //
            );
        }
        return unifiedLayer;
    }

}
