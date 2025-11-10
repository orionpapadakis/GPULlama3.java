package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;

public class F32TornadoTensor extends TornadoTensor {
    private final FloatArray values;

    public F32TornadoTensor(FloatArray values) {
        super(values.getSize());
        this.values = values;
    }

    public F32TornadoTensor(int size, MemorySegment segment) {
        super(size);
        this.values = new FloatArray(size);
        this.values.getSegment().copyFrom(segment);
    }

    @Override
    public FloatArray asFloatArray() {
        return values;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

}
