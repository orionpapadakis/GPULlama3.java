package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.GridScheduler;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal base with common fields/utilities so subclasses compile cleanly.
 * Adjust or remove fields if they already exist in your project.
 */
abstract class AbstractLayer {

    /** Optional: track the "main" task graph for the layer if one exists. */
    protected TaskGraph taskGraph;

    /** Shared runtime objects (exposed because kernels expect them). */
    protected final State state;
    protected final Weights weights;
    protected final Configuration config;

    /** Often a small context/config buffer passed into kernels. Use your real type if available. */
    protected final Object context = new Object();

    /** Common constants used in tasks & worker-grid sizing. */
    protected static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;
    protected static final int THREAD_SCALE_FOR_LOGITS = 1;

    /** Collected snapshots for scheduling / debugging. */
    protected final List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

    AbstractLayer(String taskGraphName, State state, Weights weights, Configuration config) {
        this.taskGraph = null;
        this.state = state;
        this.weights = weights;
        this.config = config;
    }

    abstract GridScheduler getGridScheduler();

    abstract TaskGraph getTaskGraph();

    abstract ImmutableTaskGraph getImmutableTaskGraph();

    /** Allow subclasses to override if they need custom transfers. */
    protected TaskGraph configureLayerDataTransfers(TaskGraph tg, int layerIndex) {
        return tg;
    }
}
