package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class Activation extends AbstractLayer{
    private final TaskGraph activationUpdate;

    public Activation(String taskGraphHandle,  State state, Weights weights, Configuration config) {
        super(taskGraphHandle, state, weights, config);

        // formatter:off
         this.activationUpdate = new TaskGraph(taskGraphHandle)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                .persistOnDevice(state.wrapX);
        // formatter:on
    }

    @Override
    GridScheduler getGridScheduler() {
        return null;
    }

    @Override
    TaskGraph getTaskGraph() {
        return activationUpdate;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return activationUpdate.snapshot();
    }

}

