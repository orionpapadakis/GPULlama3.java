package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.lang.foreign.MemorySegment;

public class FP16TornadoTensor extends TornadoTensor {
    private final HalfFloatArray values;

    public FP16TornadoTensor(int size, MemorySegment segment) {
        super(size);
        this.values = new HalfFloatArray(size);
        this.values.getSegment().copyFrom(segment);
    }

    @Override
    public HalfFloatArray asHalfFloatArray() {
        return values;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }
}

