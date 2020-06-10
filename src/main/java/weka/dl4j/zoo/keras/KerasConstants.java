package weka.dl4j.zoo.keras;

import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Map;

public class KerasConstants {

    public static Map<String, String> Locations;
    public static Map<String, Long> Checksums;

    static {
        Map<String, String> locations = new HashMap<>();
        Map<String, Long> checksums = new HashMap<>();

        // ########
        // DenseNet
        // ########
        locations.put("KerasDenseNet121", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasDenseNet121.zip");
        locations.put("KerasDenseNet169", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasDenseNet169.zip");
        locations.put("KerasDenseNet201", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasDenseNet201.zip");
        checksums.put("KerasDenseNet121", 3317961040L);
        checksums.put("KerasDenseNet169", 2917284844L);
        checksums.put("KerasDenseNet201", 390469819L);

        // ############
        // EfficientNet
        // ############
        locations.put("KerasEfficientNetB0", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB0.zip");
        locations.put("KerasEfficientNetB1", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB1.zip");
        locations.put("KerasEfficientNetB2", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB2.zip");
        locations.put("KerasEfficientNetB3", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB3.zip");
        locations.put("KerasEfficientNetB4", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB4.zip");
        locations.put("KerasEfficientNetB5", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB5.zip");
        locations.put("KerasEfficientNetB6", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB6.zip");
        locations.put("KerasEfficientNetB7", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-0.3/KerasEfficientNetB7.zip");
        checksums.put("KerasEfficientNetB0", 3915144300L);
        checksums.put("KerasEfficientNetB1", 2399258563L);
        checksums.put("KerasEfficientNetB2", 3826625376L);
        checksums.put("KerasEfficientNetB3", 4091966028L);
        checksums.put("KerasEfficientNetB4", 4208479803L);
        checksums.put("KerasEfficientNetB5", 2859282668L);
        checksums.put("KerasEfficientNetB6", 1628985300L);
        checksums.put("KerasEfficientNetB7", 1160697645L);

        // ###########
        // InceptionV3
        // ###########
        locations.put("KerasInceptionV3", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasInceptionV3.zip");
        checksums.put("KerasInceptionV3", 2527243775L);

        // ######
        // NASNet
        // ######
        locations.put("KerasNASNetMobile", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasNASNetMobile.zip");
        locations.put("KerasNASNetLarge", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasNASNetLarge.zip");
        checksums.put("KerasNASNetMobile", 1844533549L);
        checksums.put("KerasNASNetLarge", 3178325491L);

        // ######
        // ResNet
        // ######
        locations.put("KerasResNet50", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasResNet50.zip");
        locations.put("KerasResNet50V2", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasResNet50V2.zip");
        locations.put("KerasResNet101", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasResNet101.zip");
        locations.put("KerasResNet101V2", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasResNet101V2.zip");
        locations.put("KerasResNet152", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasResNet152.zip");
        locations.put("KerasResNet152V2", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasResNet152V2.zip");
        checksums.put("KerasResNet50", 3286468754L);
        checksums.put("KerasResNet50V2", 3513024472L);
        checksums.put("KerasResNet101", 220275721L);
        checksums.put("KerasResNet101V2", 3095544222L);
        checksums.put("KerasResNet152", 3085881774L);
        checksums.put("KerasResNet152V2", 2370199264L);

        // ###
        // VGG
        // ###
        locations.put("KerasVGG16", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasVGG16.zip");
        locations.put("KerasVGG19", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasVGG19.zip");
        checksums.put("KerasVGG16", 205199086L);
        checksums.put("KerasVGG19", 1716706036L);

        // ########
        // Xception
        // ########
        locations.put("KerasXception", "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasXception.zip");
        checksums.put("KerasXception", 3786331101L);

        Locations = Collections.unmodifiableMap(locations);
        Checksums = Collections.unmodifiableMap(checksums);
    }
}
