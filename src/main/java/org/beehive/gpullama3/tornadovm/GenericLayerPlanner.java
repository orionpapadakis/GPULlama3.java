package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.List;

public interface GenericLayerPlanner {

    Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered();

    Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia();

    List<ImmutableTaskGraph> getCachedTaskGraphs();

    GridScheduler getCachedGridScheduler();

}
