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
        locations.put("KerasEfficientNetB0", "https://drive.google.com/uc?export=download&id=1osvsogvbEKyT0VIZ9JtZhfgLQKCiPU8U");
        locations.put("KerasEfficientNetB1", "https://drive.google.com/uc?export=download&id=1VpMDTrSYkUs-6-Ug6FCrBX_G2WVbXoNp");
        locations.put("KerasEfficientNetB2", "https://drive.google.com/uc?export=download&id=1omrSGk5x2_lTfPSGjfIzpYlr8cU4WPbi");
        locations.put("KerasEfficientNetB3", "https://drive.google.com/uc?export=download&id=1e0gaLQ6vsCqe4mMHYDaNkd5jafXfS7HZ");
        locations.put("KerasEfficientNetB4", "https://drive.google.com/uc?export=download&id=1NDqrSBZ-p6Ct70ZwbBsZOL3YAAN-0ypz");
        locations.put("KerasEfficientNetB5", "https://drive.google.com/uc?export=download&id=1p6Hd5GNZn0R-xf4LXumFIYYAL2Jbu69y");
        locations.put("KerasEfficientNetB6", "https://drive.google.com/uc?export=download&id=1AoZunscMwqq7UhpirxfMyRMeN10NYaG6");
        locations.put("KerasEfficientNetB7", "https://drive.google.com/uc?export=download&id=1IvejTyjhXbPWDuPW5HyogwD2dv0e2Nvg");
        checksums.put("KerasEfficientNetB0", 3990389537L);
        checksums.put("KerasEfficientNetB1", 2300415769L);
        checksums.put("KerasEfficientNetB2", 2993104358L);
        checksums.put("KerasEfficientNetB3", 1778527599L);
        checksums.put("KerasEfficientNetB4", 3090591408L);
        checksums.put("KerasEfficientNetB5", 620292868L);
        checksums.put("KerasEfficientNetB6", 3557280118L);
        checksums.put("KerasEfficientNetB7", 1866748858L);

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
