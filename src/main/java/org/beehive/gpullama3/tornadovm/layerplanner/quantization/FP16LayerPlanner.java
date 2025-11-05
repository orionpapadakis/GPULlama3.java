package org.beehive.gpullama3.tornadovm.layerplanner.quantization;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.FP16Weights.FP16Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.base.QuantizedLayerPlanner;

/**
 * Base for all FP16-quantized layer planners.
 *
 * Subclasses: LlamaFP16LayerPlanner, Qwen2FP16LayerPlanner, etc.
 *
 * FP16 Specific: - Uses half-precision floating point kernels - Weights: weights.xxxHalfFloat arrays - Compute: 2x faster than FP32 on modern GPUs
 */
public abstract class FP16LayerPlanner<S extends State, C extends Configuration, W extends FP16Weights> extends QuantizedLayerPlanner<S, C, W> {

    protected FP16LayerPlanner(S state, Model model) {
        super(state, model);
        initializeLayerComponents();
    }

    @Override
    protected void validateQuantizationType() {
        // Verify we have FP16 weights
        if (this.weights.getWeightType() != GGMLType.F16) {
            throw new IllegalArgumentException("FP16LayerPlanner requires GGMLType.F16, got: " + this.weights.getWeightType());
        }
    }

    @Override
    protected void initializeLayerComponents() {
        // Override in subclasses (LlamaFP16LayerPlanner, Qwen2FP16LayerPlanner)
    }

    // FP16-specific helper methods can go here
}