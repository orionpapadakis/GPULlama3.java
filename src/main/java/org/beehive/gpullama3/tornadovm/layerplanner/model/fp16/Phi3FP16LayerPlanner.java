package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.Phi3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Phi3FP16FFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Phi3FP16LayerPlanner: Phi3 model with FP16 weights.
 *
 * Follows the same pattern as Qwen3FP16LayerPlanner but with:
 * - Phi3-specific FFN layers (combined QKV + gate/up FFN)
 * - Phi3TornadoWeights
 * - Phi3Configuration
 *
 * Inherits from FP16LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeights>
 */
public class Phi3FP16LayerPlanner extends FP16LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeights> {

    private Activation activationLayer;
    private Phi3FP16FFNLayers ffnLayers;
    private LogitsFP16Layer logitsLayer;

    // Cache
    private List<ImmutableTaskGraph> cachedTaskGraphs;
    private GridScheduler cachedScheduler;

    public Phi3FP16LayerPlanner(Phi3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);

        this.ffnLayers = new Phi3FP16FFNLayers("phi3FFN", this.state, this.weights, this.config);

        this.logitsLayer = new LogitsFP16Layer("phi3Logits", this.state, this.weights, this.config,
                ffnLayers.getLastTaskGraphID());
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() {
        if (this.cachedTaskGraphs != null && this.cachedScheduler != null) {
            return new Tuple2<>(this.cachedTaskGraphs, this.cachedScheduler);
        }

        List<ImmutableTaskGraph> allTaskGraphs = new ArrayList<>();
        GridScheduler masterScheduler = new GridScheduler();

        // 1. Activation layer
        allTaskGraphs.add(activationLayer.getImmutableTaskGraph());
        activationLayer.updateGridScheduler(masterScheduler);

        // 2. FFN layers (N transformer layers)
        allTaskGraphs.addAll(ffnLayers.getFfnLayerTaskGraphs());
        ffnLayers.updateGridScheduler(masterScheduler);

        // 3. Logits layer
        allTaskGraphs.add(logitsLayer.getTaskGraph().snapshot());
        logitsLayer.updateGridScheduler(masterScheduler);

        // Cache
        this.cachedTaskGraphs = allTaskGraphs;
        this.cachedScheduler = masterScheduler;

        return new Tuple2<>(allTaskGraphs, masterScheduler);
    }

    public void setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> allTaskGraphs = new ArrayList<>();
        GridScheduler masterScheduler = new GridScheduler();

        // 1. Activation layer
        allTaskGraphs.add(activationLayer.getImmutableTaskGraph());
        activationLayer.updateGridScheduler(masterScheduler);

        // 2. FFN layers (N transformer layers)
        allTaskGraphs.addAll(ffnLayers.getFfnLayerTaskGraphs());
        ffnLayers.updateGridScheduler(masterScheduler);

        // 3. Logits layer
        allTaskGraphs.add(logitsLayer.getTaskGraph().snapshot());
        logitsLayer.updateGridScheduler(masterScheduler);

        // Cache
        this.cachedTaskGraphs = allTaskGraphs;
        this.cachedScheduler = masterScheduler;
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
        // For now, same as NVIDIA version
        // Hardware strategy will optimize scheduler
        return setupTornadoForwardPlanLayered();
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
