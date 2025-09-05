package org.beehive.gpullama3.api.service;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.function.IntConsumer;

@Service
public class LLMService {

    @Autowired
    private ModelInitializationService initService;

    @Autowired
    private TokenizerService tokenizerService;

    public CompletableFuture<String> generateCompletion(
            String prompt,
            int maxTokens,
            double temperature,
            double topP,
            List<String> stopSequences) {

        return CompletableFuture.supplyAsync(() -> {
            try {
                System.out.println("Starting completion generation...");
                System.out.println("Prompt: " + prompt.substring(0, Math.min(50, prompt.length())) + "...");
                System.out.println("Max tokens: " + maxTokens + ", Temperature: " + temperature);

                // Get initialized components
                Model model = initService.getModel();

                // Convert prompt to tokens
                List<Integer> promptTokens = tokenizerService.encode(prompt);
                System.out.println("Prompt tokens: " + promptTokens.size());

                // Convert stop sequences to token sets
                Set<Integer> stopTokens = new HashSet<>();
                if (stopSequences != null) {
                    for (String stop : stopSequences) {
                        stopTokens.addAll(tokenizerService.encode(stop));
                    }
                    System.out.println("Stop tokens: " + stopTokens.size());
                }

                // Create custom sampler with request-specific parameters
                //Sampler sampler = initService.createCustomSampler(temperature, topP, System.currentTimeMillis());
                Sampler sampler = initService.getSampler();

                // Create state based on model type
                State state = createStateForModel(model);

                // Generate tokens using your existing method
                List<Integer> generatedTokens = model.generateTokens(
                        state,
                        0,
                        promptTokens,
                        stopTokens,
                        maxTokens,
                        sampler,
                        false,
                        token -> {} // No callback for non-streaming
                );

                // Decode tokens back to text
                String result = tokenizerService.decode(generatedTokens);
                System.out.println("Generated " + generatedTokens.size() + " tokens");
                System.out.println("Completion finished successfully");

                return result;

            } catch (Exception e) {
                System.err.println("Error generating completion: " + e.getMessage());
                e.printStackTrace();
                throw new RuntimeException("Error generating completion", e);
            }
        });
    }

    public void generateStreamingCompletion(
            String prompt,
            int maxTokens,
            double temperature,
            double topP,
            List<String> stopSequences,
            SseEmitter emitter) {

        CompletableFuture.runAsync(() -> {
            try {
                System.out.println("Starting streaming completion generation...");

                Model model = initService.getModel();

                List<Integer> promptTokens = tokenizerService.encode(prompt);

                Set<Integer> stopTokens = new HashSet<>();
                if (stopSequences != null) {
                    for (String stop : stopSequences) {
                        stopTokens.addAll(tokenizerService.encode(stop));
                    }
                }

                //Sampler sampler = initService.createCustomSampler(temperature, topP, System.currentTimeMillis());
                Sampler sampler = initService.getSampler();
                State state = createStateForModel(model);

                final int[] tokenCount = {0};

                // Streaming callback
                IntConsumer tokenCallback = token -> {
                    try {
                        String tokenText = tokenizerService.decode(List.of(token));
                        tokenCount[0]++;

                        String eventData = String.format(
                                "data: {\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":null}]}\n\n",
                                escapeJson(tokenText)
                        );

                        emitter.send(SseEmitter.event().data(eventData));

                        if (tokenCount[0] % 10 == 0) {
                            System.out.println("Streamed " + tokenCount[0] + " tokens");
                        }

                    } catch (Exception e) {
                        System.err.println("Error in streaming callback: " + e.getMessage());
                        emitter.completeWithError(e);
                    }
                };

                model.generateTokens(state, 0, promptTokens, stopTokens, maxTokens, sampler, false, tokenCallback);

                // Send completion event
                emitter.send(SseEmitter.event().data("data: [DONE]\n\n"));
                emitter.complete();

                System.out.println("Streaming completion finished. Total tokens: " + tokenCount[0]);

            } catch (Exception e) {
                System.err.println("Error in streaming generation: " + e.getMessage());
                e.printStackTrace();
                emitter.completeWithError(e);
            }
        });
    }

    /**
     * Create appropriate State subclass based on the model type
     */
    private State createStateForModel(Model model) {
        try {
            return model.createNewState();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create state for model", e);
        }
    }

    private String escapeJson(String str) {
        if (str == null) return "";
        return str.replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
                .replace("\\", "\\\\");
    }
}