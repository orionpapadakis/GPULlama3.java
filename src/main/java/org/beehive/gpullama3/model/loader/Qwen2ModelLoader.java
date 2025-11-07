package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.fp16.Qwen2TornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.LlamaTornadoWeightsQ8_0;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.Qwen2TornadoWeightsQ8_0;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.qwen2.Qwen2;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tokenizer.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.core.model.GGMLType.F16;
import static org.beehive.gpullama3.model.loader.ModelLoader.*;
import static org.beehive.gpullama3.tokenizer.Vocabulary.loadQwen3Vocabulary;

public class Qwen2ModelLoader extends AbstractModelLoader<Qwen2, Qwen2Configuration> {

    public Qwen2ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
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
    protected Qwen2Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen2.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;

        int numberOfKeyValueHeads = metadata.containsKey("qwen2.attention.head_count_kv") ? (int) metadata.get("qwen2.attention.head_count_kv") : (int) metadata.get("qwen2.attention.head_count");
        int vocabSize = vocabulary.size();

        return new Qwen2Configuration(
                (int) metadata.get("qwen2.embedding_length"),       // dim
                (int) metadata.get("qwen2.feed_forward_length"),    // hiddendim
                (int) metadata.get("qwen2.block_count"),            // numberOfLayers
                (int) metadata.get("qwen2.attention.head_count"),   // numberOfHeads

                numberOfKeyValueHeads, // numberOfKeyValueHeads
                numberOfKeyValueHeads, // numberOfHeadsKey
                numberOfKeyValueHeads, // numberOfHeadsValue

                vocabSize,
                modelContextLength,
                finalContextLength,
                false,
                (float) metadata.get("qwen2.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("qwen2.rope.freq_base")
        );
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen2Configuration config) {
        return RoPE.precomputeFreqsCis(config.contextLengthModel(), config.headSize(), config.ropeTheta(), false, 8, 1, 3, 8192);
    }

    @Override
    protected Qwen2 createModel(Qwen2Configuration config, Tokenizer tokenizer, Weights weights) {
        Map<String, Object> metadata = gguf.getMetadata();
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        // Qwen2.5-Coder uses <|endoftext|> as stop-token.
        ChatTokens chatTokens = isDeepSeekR1DistillQwen ? new ChatTokens("<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "")
                : new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen2(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Qwen2StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.bias")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadQuantized(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadQuantized(outputWeight),
                outputWeight.ggmlType()
        );
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + F16 + ")");
        }

        GGMLType ggmlType = outputWeight.ggmlType();
        return switch(ggmlType) {
            case F16 -> createTornadoVMWeightsF16(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
            case Q8_0 -> createTornadoVMWeightsQ8_0(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
            default ->
                    throw new UnsupportedOperationException("Type: " + ggmlType + " currently not supported for TornadoVM weights.");
        };
    }

    private Weights createTornadoVMWeightsF16(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                              GGMLTensorEntry outputWeight) {
        return new Qwen2TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                // Qwen2-specific: qkv bias
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.bias")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
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

    public LlamaTornadoWeightsQ8_0 createTornadoVMWeightsQ8_0(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                                  GGMLTensorEntry outputWeight) {
        return new Qwen2TornadoWeightsQ8_0(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                // Qwen2-specific: qkv bias
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.bias")),

                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
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
    private FloatTensor[] loadLayerWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, String layerName, String suffix) {
        FloatTensor[] weights = new FloatTensor[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadQuantized(tensorEntries.get(key));
        }
        return weights;
    }

    private FloatArray[] loadLayerWeightsAsFloatArraysFromBuffer(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, String layerName, String suffix) {
        FloatArray[] weights = new FloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.floatBufferToFloatArray(tensorEntries.get(key));
        }
        return weights;
    }

    private HalfFloatArray[] loadLayerWeightsAsHalfFloatArrays(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, String layerName, String suffix) {
        HalfFloatArray[] weights = new HalfFloatArray[config.numberOfLayers()];
        for (int i = 0; i < config.numberOfLayers(); i++) {
            String key = String.format("blk.%d.%s.%s", i, layerName, suffix);
            weights[i] = ModelLoader.loadTensorAsHalfFloatArray(tensorEntries.get(key));
        }
        return weights;
    }
}
