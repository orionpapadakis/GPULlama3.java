package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsLayer extends AbstractLayer{

    private TaskGraph logitsTaskGraph;
    private ImmutableTaskGraph immutableLogitsGraph;
    private GridScheduler scheduler;

    public LogitsLayer(String name, State state, Weights weights, Configuration config) {
        super(name, state, weights, config);

        if (!(weights instanceof LlamaTornadoWeights llamaWeights)) {
            throw new IllegalArgumentException("LogitsLayer requires LlamaTornadoWeights");
        }

        GGMLType wt = weights.getWeightType();
        switch (wt) {
            case F16, Q8_0 -> {
                setupLogitsTaskGraph(llamaWeights, config);
                this.scheduler = setupGridSchedulerForLogits(config);
            }
            default -> throw new UnsupportedOperationException(
                    "Quantization format " + wt + " not supported in LogitsLayer");
        }
    }

    /**
     * Builds the logits computation graph.
     */
    private void setupLogitsTaskGraph(LlamaTornadoWeights weights, Configuration config) {

        // Build logits task graph
        TaskGraph logits = new TaskGraph("logits")
                // Consume the final normalized hidden state
                .consumeFromDevice("layer_" + (config.numberOfLayers() - 1),
                        state.wrapX
                )
                // Temporary scratch buffer for RMSNorm
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits
                )
                // Transfer weights and output buffer
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapLogits,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                // Apply RMSNorm before logits projection
                .task("reductionsOneBlockLogits",
                        TransformerComputeKernels::reductionOneBlockWithLayer,
                        context, state.tempLogits,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextLogits",
                        TransformerComputeKernels::reductionOneBlock2WithLogits,
                        context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);

        // Configure quantized/fp16 matrix-vector multiply
        logits = configureQuantizedMatrixVectorFinalWeight(logits, weights, config);

        // Copy logits back to host
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);

        // Save references
        this.logitsTaskGraph = logits;
        this.immutableLogitsGraph = logits.snapshot();
        this.taskGraphs.add(this.immutableLogitsGraph);
    }

    /**
     * Selects correct kernel for final projection depending on quantization.
     */
    private TaskGraph configureQuantizedMatrixVectorFinalWeight(
            TaskGraph logits, LlamaTornadoWeights weights, Configuration config) {

        switch (weights.getWeightType()) {
            case F16 -> {
                logits.task("logits.projection",
                        TransformerComputeKernels::matrixVectorGeneric,
                        context,
                        state.wrapX, state.wrapLogits,
                        weights.wclsHalfFloat,
                        config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC);
            }
            case Q8_0 -> {
                logits.task("logits.projection",
                        TransformerComputeKernels::matrixVectorQuantized,
                        context,
                        state.wrapX, state.wrapLogits,
                        weights.wclsHalfFloat,
                        config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC);
            }
            default -> throw new UnsupportedOperationException(
                    "Unsupported logits quantization type: " + weights.getWeightType());
        }

        return logits;
    }

    private GridScheduler setupGridSchedulerForLogits(Configuration config) {
        GridScheduler scheduler = new GridScheduler();

        // RMSNorm operations
        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);
        rmsNormWorker.setLocalWork(256, 1, 1);

        // Projection kernel (vocabulary size × hidden dim)
        int vocabSizeGlobal = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        WorkerGrid projectionWorker = new WorkerGrid1D(vocabSizeGlobal);
        projectionWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);

        scheduler.addWorkerGrid("logits.projection", projectionWorker);
        scheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
        scheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

        return scheduler;
    }


    @Override
    GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    TaskGraph getTaskGraph() {
        return logitsTaskGraph;
    }

    @Override
    ImmutableTaskGraph getImmutableTaskGraph() {
        return immutableLogitsGraph;
    }
}
