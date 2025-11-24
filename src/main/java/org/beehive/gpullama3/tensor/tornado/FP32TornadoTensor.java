package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;

public class FP32TornadoTensor extends TornadoTensor {
    private final FloatArray tornadoNativeArray;

    public FP32TornadoTensor(FloatArray floatArray) {
        this.tornadoNativeArray = floatArray;
    }

    public FP32TornadoTensor(MemorySegment segment) {
        this.tornadoNativeArray = new FloatArray(segment);
    }

    @Override
    public FloatArray asFloatArray() {
        return tornadoNativeArray;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

}
