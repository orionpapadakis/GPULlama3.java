package org.beehive.gpullama3.inference.weights.tornado.q8_0;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.Q8_0QuantizedTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;


public class Qwen2TornadoWeightsQ8_0 extends LlamaTornadoWeightsQ8_0 {

    // Qwen2-specific tornado weights
    public FloatArray[] q_biasLayered;
    public FloatArray[] k_biasLayered;
    public FloatArray[] v_biasLayered;

    public Qwen2TornadoWeightsQ8_0(FloatArray tokenEmbeddingTable, FloatArray[] rms_att_weightLayered, Q8_0QuantizedTensor[] wqLayered, Q8_0QuantizedTensor[] wkLayered, Q8_0QuantizedTensor[] wvLayered,
                                   FloatArray[] wqBiasLayered,
                                   FloatArray[] wkBiasLayered,
                                   FloatArray[] wvBiasLayered,
                                   Q8_0QuantizedTensor[] woLayered, FloatArray[] rms_ffn_weightLayered, Q8_0QuantizedTensor[] w1Layered,
                                   Q8_0QuantizedTensor[] w2Layered, Q8_0QuantizedTensor[] w3Layered, FloatArray rms_final_weight_as_floatArray, FloatArray freq_cis_realFlat, FloatArray freq_cis_imagFlat, Q8_0QuantizedTensor wclsByteArray,
                                   GGMLType weightType) {
        // call to FP16Weights constructor
        super(tokenEmbeddingTable,
                rms_att_weightLayered,
                wqLayered,
                wkLayered,
                wvLayered,
                woLayered,
                rms_ffn_weightLayered,
                w1Layered,
                w2Layered,
                w3Layered,
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);
        // init qwen2-specific fields
        this.q_biasLayered = wqBiasLayered;
        this.k_biasLayered = wkBiasLayered;
        this.v_biasLayered = wvBiasLayered;
    }
}
