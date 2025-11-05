package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.Timer;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.Q8_0Weights;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.Qwen3Q8_0TornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.fp16.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.model.qwen3.Qwen3;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tokenizer.impl.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;
import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadQwen3Vocabulary;

public class Qwen3ModelLoader extends AbstractModelLoader<Qwen3, Qwen3Configuration> {

    public Qwen3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return loadQwen3Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        return new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);
    }

    @Override
    protected Qwen3Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen3.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;

        int vocabSize = vocabulary.size();

        return new Qwen3Configuration(
                (int) metadata.get("qwen3.embedding_length"),
                (int) metadata.get("qwen3.feed_forward_length"),
                (int) metadata.get("qwen3.block_count"),
                (int) metadata.get("qwen3.attention.head_count"),

                metadata.containsKey("qwen3.attention.head_count_kv") ?
                        (int) metadata.get("qwen3.attention.head_count_kv") :
                        (int) metadata.get("qwen3.attention.head_count"),
                (int) metadata.get("qwen3.attention.key_length"),
                (int) metadata.get("qwen3.attention.value_length"),

                vocabSize,
                modelContextLength,
                finalContextLength,
                false,
                (float) metadata.get("qwen3.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("qwen3.rope.freq_base")
        );
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen3Configuration config) {
        return RoPE.precomputeFreqsCis(config.contextLengthModel(), config.numberOfHeadsKey(), config.ropeTheta(), false, 0, 0, 0, 0);
    }

    @Override
    protected Qwen3 createModel(Qwen3Configuration config, Tokenizer tokenizer, Weights weights) {
        Map<String, Object> metadata = gguf.getMetadata();
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        // Qwen2.5-coder uses <|endoftext|> as stop-token.
        ChatTokens chatTokens = isDeepSeekR1DistillQwen ? new ChatTokens("<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "")
                : new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen3(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();
        return new Qwen3StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // rms_att_weight
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),       // wq
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),       // wk
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),       // wv
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),  // wo
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),  // attnKNorm
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),  // attnQNorm
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     //rms_ffn_weight
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),     // w1
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),     // w2
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),       // w3
                loadQuantized(tensorEntries.get("output_norm.weight")), // rms_final_weight
                new ArrayFloatTensor(ropeFreqsReal),
                new ArrayFloatTensor(ropeFreqsImag),
                tensorEntries.containsKey("output.weight")
                        ? ModelLoader.loadQuantized(tensorEntries.get("output.weight"))
                        : loadQuantized(tokenEmbeddings), // weights are shared
                null
        );
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                             GGMLTensorEntry outputWeight) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
        }

        GGMLType ggmlType = outputWeight.ggmlType();
        return switch(ggmlType) {
            case F16 -> createTornadoVMWeightsF16(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
            case Q8_0 -> createTornadoVMWeightsQ8_0(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
            default -> throw new UnsupportedOperationException("Type: " + ggmlType + " currently not supported for TornadoVM weights.");
        };

    }

    private Weights createTornadoVMWeightsF16(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                              GGMLTensorEntry outputWeight) {
        return new Qwen3TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),   // attnKNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),   // attnQNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),            // w1
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),            // w2
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),              // w3
                floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()),
                FloatArray.fromArray(ropeFreqs.second()),
                loadTensorAsHalfFloatArray(outputWeight),
                outputWeight.ggmlType()
        );
    }

    private Q8_0Weights createTornadoVMWeightsQ8_0(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                                   GGMLTensorEntry outputWeight) {
        return new Qwen3Q8_0TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),   // attnKNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),   // attnQNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),            // w1
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),            // w2
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),              // w3
                floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()),
                FloatArray.fromArray(ropeFreqs.second()),
                loadQ8_0QuantizedTensor(outputWeight),
                outputWeight.ggmlType()
        );
    }

    // Helper methods
    private FloatTensor[] loadLayerWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, String layerName, String suffix) {
        FloatTensor[] weights = new FloatTensor[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadQuantized(tensorEntries.get(key));
        }
        return weights;
    }

    private FloatArray[] loadLayerWeightsAsFloatArraysFromBuffer(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, String layerName, String suffix) {
        FloatArray[] weights = new FloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.floatBufferToFloatArray(tensorEntries.get(key));
        }
        return weights;
    }

    private HalfFloatArray[] loadLayerWeightsAsHalfFloatArrays(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config, String layerName, String suffix) {
        HalfFloatArray[] weights = new HalfFloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadTensorAsHalfFloatArray(tensorEntries.get(key));
        }
        return weights;
    }
}
