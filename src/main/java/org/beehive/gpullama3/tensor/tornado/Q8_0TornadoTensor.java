package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;

import java.lang.foreign.MemorySegment;

public class Q8_0TornadoTensor extends TornadoTensor {

    private final HalfFloatArray scales;  // One per 32-element block
    private final Int8Array quants;       // Quantized int8 values
    private MemorySegment segment;

    public Q8_0TornadoTensor(int size, HalfFloatArray scales, Int8Array quants, MemorySegment segment) {
        super(size);
        this.scales = scales;
        this.quants = quants;
        this.segment = segment;
    }

    /**
     * Returns the scale factors for GPU kernels.
     *
     * @return HalfFloatArray containing fp16 scale factors
     */
    public HalfFloatArray getScales() {
        return scales;
    }

    /**
     * Returns the quantized values for GPU kernels.
     *
     * @return Int8Array containing quantized int8 values
     */
    public Int8Array getQuants() {
        return quants;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    public MemorySegment asMemorySegment() {
        return segment;
    }

    /**
     * Dequantizes and returns a single float value.
     *
     * @param index Element index
     * @return Dequantized float value
     */
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIdx = index / GGMLType.Q8_0.getBlockSize();
        float scale = scales.get(blockIdx).getFloat32();
        byte quant = quants.get(index);
        return quant * scale;
    }
}
