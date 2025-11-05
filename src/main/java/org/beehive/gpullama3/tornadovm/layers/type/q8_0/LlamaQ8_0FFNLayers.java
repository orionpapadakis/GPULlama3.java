package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.LlamaTornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Q8_0Weights;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Qwen3Q8_0TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layers.AbstractLayer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

public class LlamaQ8_0FFNLayers extends AbstractLayer {


    String lastTaskGraphID;
    TaskGraph ffunnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    public LlamaQ8_0FFNLayers(String taskGraphName, LlamaState state, Q8_0Weights weights, Configuration config) {
        super(taskGraphName, state, weights, config);
        ffnLayerTaskGraphs = setupFFNLayered();
        this.scheduler = setupGridSchedulersLayered();
    }

    @Override
    public GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return null;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return null;
    }

    List<ImmutableTaskGraph> setupFFNLayered() {
        List<ImmutableTaskGraph> ffnGraphs = new ArrayList<>();
        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);

        for (int layerIndex =0; layerIndex < config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSingleFFNLayer((Q8_0Weights) weights, config, layerIndex);
            if ( layerIndex == config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
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
    TaskGraph setupSingleFFNLayer(Q8_0Weights weights, Configuration config, int layerIndex) {

        TaskGraph unifiedLayer = null;
            unifiedLayer = new TaskGraph("layer_" + layerIndex);
            unifiedLayer.consumeFromDevice(state.wrapX);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    //Copy-in weights per layer for batched-layered layout
                    weights.rms_att_weightLayered[layerIndex],
                    weights.wqLayered[layerIndex].getQuants(),
                    weights.wqLayered[layerIndex].getScales(),
                    weights.wkLayered[layerIndex].getQuants(),
                    weights.wkLayered[layerIndex].getScales(),
                    weights.wvLayered[layerIndex].getQuants(),
                    weights.wvLayered[layerIndex].getScales(),
                    weights.woLayered[layerIndex].getQuants(),
                    weights.woLayered[layerIndex].getScales(),
                    weights.rms_ffn_weightLayered[layerIndex],
                    weights.w1Layered[layerIndex].getQuants(),
                    weights.w1Layered[layerIndex].getScales(),
                    weights.w2Layered[layerIndex].getQuants(),
                    weights.w2Layered[layerIndex].getScales(),
                    weights.w3Layered[layerIndex].getQuants(),
                    weights.w3Layered[layerIndex].getScales()
            );
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                    .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb,  state.wrapQ, weights.wqLayered[layerIndex].getQuants(), weights.wqLayered[layerIndex].getScales(), config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb,  state.wrapK, weights.wkLayered[layerIndex].getQuants(), weights.wkLayered[layerIndex].getScales(), config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                            state.wrapXb,   state.wrapV, weights.wvLayered[layerIndex].getQuants(), weights.wvLayered[layerIndex].getScales(), config.dim(), config.kvDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("rope", TransformerComputeKernelsLayered::ropeRotation,context,
                            state.positionHolder, state.wrapQ, state.wrapK, config.kvDim(),
                            config.headSize())
                    .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                            state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength())
                    .task("parallel-attention", TransformerComputeKernelsLayered::processHeadsFlashAttention, context,
                            state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                            config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                            state.positionHolder, layerIndex, config.contextLength())
                    .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapXb,  state.wrapX, weights.woLayered[layerIndex].getQuants(), weights.woLayered[layerIndex].getScales(), config.dim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                            state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                    .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                            state.wrapXb,   state.wrapHb, weights.w1Layered[layerIndex].getQuants(), weights.w1Layered[layerIndex].getScales(), weights.w3Layered[layerIndex].getQuants(), weights.w3Layered[layerIndex].getScales(), config.dim(), config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                            state.wrapHb, state.wrapX, weights.w2Layered[layerIndex].getQuants(), weights.w2Layered[layerIndex].getScales(), config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                    .persistOnDevice(
                            state.wrapX
                    );
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

    private GridScheduler setupGridSchedulersLayered() {
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
        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim() / 2);
        ropeWorker.setGlobalWork(config.dim() / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

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

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
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
        WorkerGrid ropeWorker = new WorkerGrid1D(config.dim() / 2);
        ropeWorker.setGlobalWork(config.dim() / 2, 1, 1);
        ropeWorker.setLocalWork(128, 1, 1);

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

        // Map workers to tasks
        tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
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

    public List<ImmutableTaskGraph> getFfnLayerTaskGraphs() {
        return ffnLayerTaskGraphs;
    }

}
