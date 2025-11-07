package org.beehive.gpullama3.core.model.tensor;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.beehive.gpullama3.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.lang.foreign.MemorySegment;

public class F16QuantizedTensor extends TornadoTensor {
    private final HalfFloatArray values;

    public F16QuantizedTensor(int size, MemorySegment segment) {
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

