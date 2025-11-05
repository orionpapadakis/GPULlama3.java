package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.FP16Weights;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layers.AbstractLayer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsFP16Layer extends AbstractLayer {

    private String lastTaskGraphID;
    private TaskGraph logitsTaskGraph;
    private ImmutableTaskGraph immutableLogitsGraph;
    private GridScheduler scheduler;

    public LogitsFP16Layer(String name, State state, Weights weights, Configuration config, String lastTaskGraphID) {
        super(name, state, weights, config);
        this.lastTaskGraphID = lastTaskGraphID;
        state.tempLogits.init(0.0f);

        if (!(weights instanceof FP16Weights fp16Weights )) {
            throw new IllegalArgumentException("LogitsLayer requires LlamaTornadoWeights");
        }

        this.logitsTaskGraph = setupLogitsTaskGraph(fp16Weights , config);
    }

    private TaskGraph setupLogitNonNVidia(FP16Weights weights, Configuration config) {
        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(lastTaskGraphID,
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapLogits,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);
        logits.task("projection", TransformerComputeKernelsLayered::matrixVectorGeneric,  //
                context, state.wrapX, state.wrapLogits, weights.wclsHalfFloat, //
                config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS); //
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        return logits;
    }

    /**
     * Builds the logits computation graph.
     */
    private TaskGraph setupLogitsTaskGraph(FP16Weights weights, Configuration config) {

        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(lastTaskGraphID,
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapLogits,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);
                logits.task("projection", TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context, state.wrapX, state.wrapLogits, weights.wclsHalfFloat,
                        config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);
                logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);

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

//    @Override
//    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
//        // RMSNorm operations
//        WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
//        rmsNormWorker.setGlobalWork(config.dim(), 1, 1);
//        rmsNormWorker.setLocalWork(256, 1, 1);
//
//        // Projection kernel (vocabulary size × hidden dim)
//        int vocabSizeGlobal = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
//        WorkerGrid projectionWorker = new WorkerGrid1D(vocabSizeGlobal);
//        projectionWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);
//
//        scheduler.addWorkerGrid("logits.projection", projectionWorker);
//        scheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
//        scheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);
//
//        return scheduler;
//    }


    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
            // RMSNorm operations
            WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
            rmsNormWorker.setGlobalWork(config.dim(), 1, 1);
            rmsNormWorker.setLocalWork(256, 1, 1);

        // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.vocabularySize,1,1], localWorkSize=[16,1,1])
        // CUDA equivalent: kernel<<<dim3((config.vocabularySize+15)/16,1,1), dim3(16,1,1)>>>
        int vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        WorkerGrid vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);

        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);
        return scheduler;
    }

    @Override
    public GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return logitsTaskGraph;
    }

    @Override
    public  ImmutableTaskGraph getImmutableTaskGraph() {
        return immutableLogitsGraph;
    }
}
