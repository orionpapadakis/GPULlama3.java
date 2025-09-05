package org.beehive.gpullama3.api.service;

import org.beehive.gpullama3.model.Model;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class TokenizerService {

    @Autowired
    private ModelInitializationService initService;

    public List<Integer> encode(String text) {
        Model model = initService.getModel();
        // Use your model's tokenizer - adapt this to your actual tokenizer interface
        // This assumes your Model has a tokenizer() method that returns a Tokenizer
        return model.tokenizer().encodeAsList(text);
    }

    public String decode(List<Integer> tokens) {
        Model model = initService.getModel();
        // Use your model's tokenizer for decoding
        return model.tokenizer().decode(tokens);
    }

//    public String decode(int token) {
//        Model model = initService.getModel();
//        // Convenience method for single token decoding
//        return model.tokenizer().decode(token);
//    }
}