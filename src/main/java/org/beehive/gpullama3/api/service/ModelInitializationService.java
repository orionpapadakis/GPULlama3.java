package org.beehive.gpullama3.api.service;

import org.beehive.gpullama3.LlamaApp;
import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.springframework.boot.ApplicationArguments;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.IOException;

import static org.beehive.gpullama3.inference.sampler.Sampler.createSampler;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

@Service
public class ModelInitializationService {

    private final ApplicationArguments args;

    private Options options;
    private Model model;
    private Sampler sampler;

    public ModelInitializationService(ApplicationArguments args) {
        this.args = args;
    }

    @PostConstruct
    public void init() {
        try {
            System.out.println("=== Model Initialization Service ===");
            System.out.println("Initializing model service...");

            // Step 1: Parse options from command line arguments
            System.out.println("Step 1: Parsing options...");
            options = Options.parseOptions(args.getSourceArgs());
            System.out.println("✓ Options parsed successfully");

            // Step 2: Load model
            System.out.println("\nStep 2: Loading model...");
            System.out.println("Loading model from: " + options.modelPath());
            model = loadModel(options);
            System.out.println("✓ Model loaded successfully");
            System.out.println("  Model type: " + model.getClass().getSimpleName());
            System.out.println("  Vocabulary size: " + model.configuration().vocabularySize());
            System.out.println("  Context length: " + model.configuration().contextLength());

            // Step 3: Create default sampler
            System.out.println("\nStep 3: Creating default sampler...");
            sampler = createSampler(model, options);
            System.out.println("✓ Default sampler created");
            System.out.println("  Sampler type: " + sampler.getClass().getSimpleName());

            System.out.println("\n✓ Model service initialization completed successfully!");
            System.out.println("=== Ready to serve requests ===\n");

        } catch (Exception e) {
            System.err.println("✗ Failed to initialize model service: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("Model initialization failed", e);
        }
    }

    // Getters for other services to access the initialized components
    public Options getOptions() {
        if (options == null) {
            throw new IllegalStateException("Model service not initialized yet");
        }
        return options;
    }

    public Model getModel() {
        if (model == null) {
            throw new IllegalStateException("Model service not initialized yet");
        }
        return model;
    }

    public Sampler getSampler() {
        if (sampler == null) {
            throw new IllegalStateException("Model service not initialized yet");
        }
        return sampler;
    }
}