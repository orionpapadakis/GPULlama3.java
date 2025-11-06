package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.List;

public interface GenericLayerPlanner {

    List<ImmutableTaskGraph> getCachedTaskGraphs();

    GridScheduler getCachedGridScheduler();

}
