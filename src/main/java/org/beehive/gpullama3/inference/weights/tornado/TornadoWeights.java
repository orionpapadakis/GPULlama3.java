package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.inference.weights.Weights;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public interface TornadoWeights extends Weights {

    FloatArray getTokenEmbeddingTable();

}
