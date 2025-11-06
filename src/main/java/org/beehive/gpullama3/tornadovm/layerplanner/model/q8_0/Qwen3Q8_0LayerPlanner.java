package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Q8_0Weights;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Qwen3Q8_0TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Qwen3Q8_0FFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen3Q8_0LayerPlanner: Qwen3 model with Q8_0-quantized weights.
 *
 * Follows the same pattern as LlamaQ8_0LayerPlanner but with:
 * - Qwen3-specific FFN layers (supports GQA)
 * - Qwen3Q8_0TornadoWeights (8-bit integer quantization)
 * - Qwen3Configuration
 * - 2x memory compression vs FP16
 *
 * Inherits from Q8_0LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3Q8_0TornadoWeights>
 */
public class Qwen3Q8_0LayerPlanner extends Q8_0LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3Q8_0TornadoWeights> {

    private Activation activationLayer;
    private Qwen3Q8_0FFNLayers ffnLayers;
    private LogitsQ8_0Layer logitsLayer;

    // Cache
    private List<ImmutableTaskGraph> cachedTaskGraphs;
    private GridScheduler cachedScheduler;

    public Qwen3Q8_0LayerPlanner(Qwen3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);

        this.ffnLayers = new Qwen3Q8_0FFNLayers("qwen3FFN", this.state, this.weights, this.config);

        this.logitsLayer = new LogitsQ8_0Layer("qwen3Logits", this.state, this.weights, this.config,
                ffnLayers.getLastTaskGraphID());
    }

    public void setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> allTaskGraphs = new ArrayList<>();
        GridScheduler masterScheduler = new GridScheduler();

        // 1. Activation layer
        allTaskGraphs.add(activationLayer.getImmutableTaskGraph());
        activationLayer.updateGridScheduler(masterScheduler);

        // 2. FFN layers (N transformer layers with GQA support and Q8_0 quantization)
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