package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Qwen2FP16FFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen2FP16LayerPlanner: Qwen2 model with FP16 weights.
 *
 * Follows the same pattern as LlamaFP16LayerPlanner but with:
 * - Qwen2-specific FFN layers (supports GQA with bias terms)
 * - Qwen2TornadoWeights
 * - Qwen2Configuration
 *
 * Inherits from FP16LayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights>
 */
public class Qwen2FP16LayerPlanner extends FP16LayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights> {

    private Activation activationLayer;
    private Qwen2FP16FFNLayers ffnLayers;
    private LogitsFP16Layer logitsLayer;

    // Cache
    private List<ImmutableTaskGraph> cachedTaskGraphs;
    private GridScheduler cachedScheduler;

    public Qwen2FP16LayerPlanner(Qwen2State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);

        this.ffnLayers = new Qwen2FP16FFNLayers("qwen2FFN", this.state, this.weights, this.config);

        this.logitsLayer = new LogitsFP16Layer("qwen2Logits", this.state, this.weights, this.config,
                ffnLayers.getLastTaskGraphID());
    }

    public void setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> allTaskGraphs = new ArrayList<>();
        GridScheduler masterScheduler = new GridScheduler();

        // 1. Activation layer
        allTaskGraphs.add(activationLayer.getImmutableTaskGraph());
        activationLayer.updateGridScheduler(masterScheduler);

        // 2. FFN layers (N transformer layers with GQA support and bias terms)
        allTaskGraphs.addAll(ffnLayers.getFfnLayerTaskGraphs());
        ffnLayers.updateGridScheduler(masterScheduler);

        // 3. Logits layer
        allTaskGraphs.add(logitsLayer.getTaskGraph().snapshot());
        logitsLayer.updateGridScheduler(masterScheduler);

        // Cache
        this.cachedTaskGraphs = allTaskGraphs;
        this.cachedScheduler = masterScheduler;
    }

    public List<ImmutableTaskGraph> getCachedTaskGraphs() {
        return this.cachedTaskGraphs;
    }

    @Override
    public GridScheduler getCachedGridScheduler() {
        return this.cachedScheduler;
    }

    public void clearCache() {
        this.cachedTaskGraphs = null;
        this.cachedScheduler = null;
    }
}
