package org.beehive.gpullama3.api.controller;

import org.beehive.gpullama3.api.model.CompletionRequest;
import org.beehive.gpullama3.api.model.CompletionResponse;
import org.beehive.gpullama3.api.service.LLMService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/v1")
@CrossOrigin(origins = "*")
public class CompletionController {

    @Autowired
    private LLMService llmService;

    @PostMapping("/completions")
    public CompletableFuture<ResponseEntity<CompletionResponse>> createCompletion(@RequestBody CompletionRequest request) {

        System.out.println("Received completion request: " + request);

        if (Boolean.TRUE.equals(request.getStream())) {
            throw new IllegalArgumentException("Use /v1/completions/stream for streaming requests");
        }

        // Validate request
        if (request.getPrompt() == null || request.getPrompt().trim().isEmpty()) {
            throw new IllegalArgumentException("Prompt cannot be null or empty");
        }

        return llmService.generateCompletion(
                request.getPrompt(),
                request.getMaxTokens(),
                request.getTemperature(),
                request.getTopP(),
                request.getStopSequences()
        ).thenApply(generatedText -> {
            CompletionResponse response = new CompletionResponse();
            response.setModel(request.getModel());

            CompletionResponse.Choice choice = new CompletionResponse.Choice(
                    generatedText, 0, "stop");
            response.setChoices(Arrays.asList(choice));

            // Calculate rough token counts (you might want to make this more accurate)
            int promptTokens = request.getPrompt().length() / 4; // Rough estimate
            int completionTokens = generatedText.length() / 4;   // Rough estimate
            CompletionResponse.Usage usage = new CompletionResponse.Usage(promptTokens, completionTokens);
            response.setUsage(usage);

            System.out.println("Completion response prepared, length: " + generatedText.length());

            return ResponseEntity.ok(response);
        });
    }

    @PostMapping(value = "/completions/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter createStreamingCompletion(@RequestBody CompletionRequest request) {

        System.out.println("Received streaming completion request: " + request);

        // Validate request
        if (request.getPrompt() == null || request.getPrompt().trim().isEmpty()) {
            throw new IllegalArgumentException("Prompt cannot be null or empty");
        }

        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);

        llmService.generateStreamingCompletion(
                request.getPrompt(),
                request.getMaxTokens(),
                request.getTemperature(),
                request.getTopP(),
                request.getStopSequences(),
                emitter
        );

        return emitter;
    }

    @GetMapping("/models")
    public ResponseEntity<Object> listModels() {
        return ResponseEntity.ok(new Object() {
            public final String object = "list";
            public final Object[] data = new Object[] {
                    new Object() {
                        public final String id = "gpullama3";
                        public final String object = "model";
                        public final long created = System.currentTimeMillis() / 1000;
                        public final String owned_by = "beehive";
                    }
            };
        });
    }

    @GetMapping("/health")
    public ResponseEntity<Object> health() {
        return ResponseEntity.ok(new Object() {
            public final String status = "healthy";
            public final long timestamp = System.currentTimeMillis();
        });
    }

    // Global exception handler for this controller
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Object> handleBadRequest(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(new Object() {
            public final String error = e.getMessage();
            public final long timestamp = System.currentTimeMillis();
        });
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Object> handleInternalError(Exception e) {
        System.err.println("Internal server error: " + e.getMessage());
        e.printStackTrace();

        return ResponseEntity.internalServerError().body(new Object() {
            public final String error = "Internal server error";
            public final String message = e.getMessage();
            public final long timestamp = System.currentTimeMillis();
        });
    }
}