package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.TornadoTensor;

public class Qwen3TornadoWeights extends TornadoWeights {
    // Qwen3-specific fields
    public final TornadoTensor[] rms_att_KNormLayered;
    public final TornadoTensor[] rms_att_QNormLayered;

    public Qwen3TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] rmsAttWeight,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] rms_att_KNormLayered,
            TornadoTensor[] rms_att_QNormLayered,
            TornadoTensor[] rmsFFNWeight,
            TornadoTensor[] w1,
            TornadoTensor[] w2,
            TornadoTensor[] w3,
            TornadoTensor rmsFinalWeight,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            TornadoTensor wCls,
            GGMLType weightType) {
        super(tokenEmbeddingTable, rmsAttWeight, wq, wk, wv, wo,
                rmsFFNWeight, w1, w2, w3, rmsFinalWeight,
                freqCisReal, freqCisImag, wCls, weightType);
        this.rms_att_KNormLayered = rms_att_KNormLayered;
        this.rms_att_QNormLayered = rms_att_QNormLayered;
    }

}
