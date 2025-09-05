package org.beehive.gpullama3;

import org.beehive.gpullama3.aot.AOT;
import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.inference.sampler.CategoricalSampler;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.sampler.ToppSampler;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.FloatArrayUtils;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import static org.beehive.gpullama3.inference.sampler.Sampler.createSampler;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

public class LlamaApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));   // Enable Java Vector API for CPU acceleration
    public static final boolean SHOW_PERF_INTERACTIVE = Boolean.parseBoolean(System.getProperty("llama.ShowPerfInteractive", "true")); // Show performance metrics in interactive mode

    private static void runSingleInstruction(Model model, Sampler sampler, Options options) {
        String response = model.runInstructOnce(sampler, options);
        System.out.println(response);
        if (SHOW_PERF_INTERACTIVE) {
            LastRunMetrics.printMetrics();
        }
    }

    /**
     * Entry point for running the LLaMA-based model with provided command-line arguments.
     *
     * <p>Initializes model options, loads the appropriate model (either AOT or on-demand),
     * configures the sampler, and runs either in interactive or single-instruction mode based on the input options.</p>
     *
     * @param args
     *         command-line arguments used to configure model path, temperature, seed, etc.
     * @throws IOException
     *         if model loading or file operations fail.
     */
    static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        Sampler sampler = createSampler(model, options);

        if (options.interactive()) {
            model.runInteractive(sampler, options);
        } else {
            runSingleInstruction(model, sampler, options);
        }
    }
}



