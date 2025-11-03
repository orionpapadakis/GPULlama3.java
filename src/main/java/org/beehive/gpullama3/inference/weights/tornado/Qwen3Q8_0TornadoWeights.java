package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.Q8_0QuantizedTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;


public class Qwen3Q8_0TornadoWeights extends Q8_0Weights{

    //attnKNorm
    public FloatArray[] rms_att_KNormLayered;
    //attnQNorm
    public FloatArray[] rms_att_QNormLayered;

    // @formatter:off
    public Qwen3Q8_0TornadoWeights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            Q8_0QuantizedTensor[] wqLayered,
            Q8_0QuantizedTensor[] wkLayered,
            Q8_0QuantizedTensor[] wvLayered,
            Q8_0QuantizedTensor[] woLayered,
            FloatArray[] rms_att_KNormLayered,
            FloatArray[] rms_att_QNormLayered,
            FloatArray[] rms_ffn_weightLayered,
            Q8_0QuantizedTensor[] w1Layered,
            Q8_0QuantizedTensor[] w2Layered,
            Q8_0QuantizedTensor[] w3Layered,
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            Q8_0QuantizedTensor wclsByteArray,
            GGMLType weightType) {
        // call to Q8_0Weights constructor
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
        // init qwen3-specific fields
        this.rms_att_KNormLayered = rms_att_KNormLayered;
        this.rms_att_QNormLayered = rms_att_QNormLayered;
    }
    // @formatter:on

}
