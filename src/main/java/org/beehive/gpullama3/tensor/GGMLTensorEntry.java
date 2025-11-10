package org.beehive.gpullama3.tensor;

import java.lang.foreign.MemorySegment;

public record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                              MemorySegment memorySegment) {
}
