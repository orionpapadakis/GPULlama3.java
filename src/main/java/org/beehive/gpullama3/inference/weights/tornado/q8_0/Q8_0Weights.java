package org.beehive.gpullama3.inference.weights.tornado.q8_0;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.Q8_0QuantizedTensor;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class Q8_0Weights implements TornadoWeights {
    public FloatArray[] rms_att_weightLayered;          // (layer, dim) rmsnorm weights
    public Q8_0QuantizedTensor[] wqLayered;                  // (layer, n_heads * head_size)
    public Q8_0QuantizedTensor[] wkLayered;                  // (layer, n_kv_heads, head_size)
    public Q8_0QuantizedTensor[] wvLayered;                  // (layer, n_kv_heads * head_size)
    public Q8_0QuantizedTensor[] woLayered;                  // (layer, n_heads * head_size, dim)
    public FloatArray[] rms_ffn_weightLayered;          // (layer, dim)
    public Q8_0QuantizedTensor[] w1Layered;                  // (layer, hidden_dim, dim)
    public Q8_0QuantizedTensor[] w2Layered;                  // (layer, dim, hidden_dim)
    public Q8_0QuantizedTensor[] w3Layered;                  // (layer, hidden_dim, dim)
    public FloatArray rms_final_weight_as_floatArray;
    public FloatArray tokenEmbeddingTable;              // (vocab_size, dim)
    public FloatArray freq_cis_realFlat;                // (seq_len, head_size/2)
    public FloatArray freq_cis_imagFlat;                // (seq_len, head_size/2)
    public Q8_0QuantizedTensor wclsHalfFloat;

    // (optional) classifier weights for the logits, on the last layer
    protected final GGMLType weightType;

    public Q8_0Weights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            Q8_0QuantizedTensor[] wqLayered,
            Q8_0QuantizedTensor[] wkLayered,
            Q8_0QuantizedTensor[] wvLayered,
            Q8_0QuantizedTensor[] woLayered,
            FloatArray[] rms_ffn_weightLayered,
            Q8_0QuantizedTensor[] w1Layered,
            Q8_0QuantizedTensor[] w2Layered,
            Q8_0QuantizedTensor[] w3Layered,
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            Q8_0QuantizedTensor wclsByteArray,
            GGMLType weightType) {
        // TornadoVM format
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.rms_att_weightLayered = rms_att_weightLayered;
        this.wqLayered = wqLayered;
        this.wkLayered = wkLayered;
        this.wvLayered = wvLayered;
        this.woLayered = woLayered;
        this.rms_ffn_weightLayered = rms_ffn_weightLayered;
        this.w1Layered = w1Layered;
        this.w2Layered = w2Layered;
        this.w3Layered = w3Layered;
        this.rms_final_weight_as_floatArray = rms_final_weight_as_floatArray;
        this.freq_cis_realFlat = freq_cis_realFlat;
        this.freq_cis_imagFlat = freq_cis_imagFlat;
        this.wclsHalfFloat = wclsByteArray;
        this.weightType = weightType;
    }
    //@formatter:on

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }

    @Override
    public FloatArray getTokenEmbeddingTable() {
        return tokenEmbeddingTable;
    }
}
