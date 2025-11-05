package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.LlamaTornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Q8_0Weights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.GPULLlama3TypeException;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

public class LlamaQ8_0LayerPlanner extends Q8_0LayerPlanner<LlamaState, LlamaConfiguration, Q8_0Weights> {

    private Activation activationLayer;
    private LlamaQ8_0FFNLayers ffnLayers;
    private LogitsQ8_0Layer logitsLayer;


    // Cache
    private List<ImmutableTaskGraph> cachedTaskGraphs;
    private GridScheduler cachedScheduler;


    public LlamaQ8_0LayerPlanner(LlamaState state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
        return null;
    }


    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);

        this.ffnLayers = new LlamaQ8_0FFNLayers("llamaFFN", this.state, this.weights, this.config);

        this.logitsLayer = new LogitsQ8_0Layer("llamaLogits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID());
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

//    @Override
//    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
//        // For now, same as NVIDIA version
//        // Hardware strategy will optimize scheduler
//        return setupTornadoForwardPlanLayered();
//    }

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