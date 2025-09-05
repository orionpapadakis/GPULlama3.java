package org.beehive.gpullama3.api.config;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.api.service.ModelInitializationService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ModelConfiguration {

    /**
     * Expose Model as a Spring bean using the initialized service
     */
    @Bean
    public Model model(ModelInitializationService initService) {
        return initService.getModel();
    }

    /**
     * Expose Options as a Spring bean using the initialized service
     */
    @Bean
    public Options options(ModelInitializationService initService) {
        return initService.getOptions();
    }
}