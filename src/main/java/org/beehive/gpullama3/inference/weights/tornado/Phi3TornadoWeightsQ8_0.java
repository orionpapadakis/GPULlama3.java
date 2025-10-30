package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.Q8_0QuantizedTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;


public class Phi3TornadoWeightsQ8_0 extends Q8_0Weights {

    // Phi3-specific weight arrays
    public Q8_0QuantizedTensor[] wqkvLayered;    // Combined QKV weights: (layer, op_size, dim) where op_size = dim + 2 * (n_kv_heads * head_dim)
    public Q8_0QuantizedTensor[] wDownLayered;   // FFN down projection: (layer, dim, hidden_dim)
    public Q8_0QuantizedTensor[] wUpLayered;     // FFN up projection: (layer, hidden_dim, dim)

    // @formatter:off
    public Phi3TornadoWeightsQ8_0(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            Q8_0QuantizedTensor[] wqkvLayered,        // Combined QKV weights for Phi3
            Q8_0QuantizedTensor[] woLayered,
            FloatArray[] rms_ffn_weightLayered,
            Q8_0QuantizedTensor[] wDownLayered,       // FFN down weights
            Q8_0QuantizedTensor[] wUpLayered,         // FFN up weights
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            Q8_0QuantizedTensor wclsByteArray,
            GGMLType weightType) {

        // Call to Q8_0Weights constructor with null values for unused standard weights
        super(tokenEmbeddingTable,
                rms_att_weightLayered,
                null,  // wqLayered - not used in Phi3, using combined wqkv instead
                null,  // wkLayered - not used in Phi3, using combined wqkv instead
                null,  // wvLayered - not used in Phi3, using combined wqkv instead
                woLayered,
                rms_ffn_weightLayered,
                null,  // w1Layered - not used in Phi3, using wUp instead
                null,  // w2Layered - not used in Phi3, using wDown instead
                null,  // w3Layered - not used in Phi3, using wUp instead
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);

        // Initialize Phi3-specific fields
        this.wqkvLayered = wqkvLayered;
        this.wDownLayered = wDownLayered;
        this.wUpLayered = wUpLayered;
    }
// @formatter:on
}
