package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.F32QuantizedTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.qwen3.Qwen3;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tokenizer.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;
import static org.beehive.gpullama3.tokenizer.Vocabulary.loadQwen3Vocabulary;

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
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen3Configuration config,
                                             Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                             GGMLTensorEntry outputWeight) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
        }

        GGMLType ggmlType = outputWeight.ggmlType();

        return new Qwen3TornadoWeights(
                loadTornadoTensorAsF32(tokenEmbeddings),
                loadArrayOfTornadoTensorsAsF32(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTornadoTensors(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensors(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensors(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTornadoTensors(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                // Qwen3-specific: attnKNorm and attnQNorm (always F32)
                loadArrayOfTornadoTensorsAsF32(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfTornadoTensorsAsF32(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfTornadoTensorsAsF32(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTornadoTensors(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensors(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensorAsF32(tensorEntries.get("output_norm.weight")),
                new F32QuantizedTensor(FloatArray.fromArray(ropeFreqs.first())),
                new F32QuantizedTensor(FloatArray.fromArray(ropeFreqs.second())),
                loadTornadoTensor(outputWeight),
                ggmlType
        );

    }
}
