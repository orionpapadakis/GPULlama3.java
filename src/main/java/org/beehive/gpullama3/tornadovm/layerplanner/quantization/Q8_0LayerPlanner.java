package org.beehive.gpullama3.tornadovm.layerplanner.quantization;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights.Q8_0Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.base.QuantizedLayerPlanner;

/**
 * Base for all Q8_0-quantized layer planners.
 *
 * Subclasses: LlamaQ8_0LayerPlanner, Qwen2Q8_0LayerPlanner, etc.
 *
 * Q8_0 Specific: - Uses 8-bit integer quantization with uniform scaling per 32-element block - Weights: weights.xxxByteArray arrays - Compute: dequantize on-the-fly during matmul - Memory: 2x
 * compression vs FP16
 */
public abstract class Q8_0LayerPlanner<S extends State, C extends Configuration, W extends Q8_0Weights> extends QuantizedLayerPlanner<S, C, W> {

    protected Q8_0LayerPlanner(S state, Model model) {
        super(state, model);
        initializeLayerComponents();
    }

    @Override
    protected void validateQuantizationType() {
        // Verify we have Q8_0 weights
        if (this.weights.getWeightType() != GGMLType.Q8_0) {
            throw new IllegalArgumentException("Q8_0LayerPlanner requires GGMLType.Q8_0, got: " + this.weights.getWeightType());
        }
    }

    @Override
    protected void initializeLayerComponents() {
        // Override in subclasses (LlamaQ8_0LayerPlanner, etc.)
    }

    // Q8_0-specific helper methods can go here
    // E.g., dequantization utilities used in compute kernels
}