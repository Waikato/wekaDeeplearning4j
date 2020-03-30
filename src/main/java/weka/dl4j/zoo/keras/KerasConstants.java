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
        locations.put("DenseNet121", "https://drive.google.com/uc?export=download&id=1skHkrjXhCv10-ASZelLaF98lAxOB9c1-");
        locations.put("DenseNet169", "https://drive.google.com/uc?export=download&id=1n6E--TOXBmsLoROzYHI7KJYpSe4BJFS5");
        locations.put("DenseNet201", "https://drive.google.com/uc?export=download&id=10Ki6JCHmB27LJw1nKs4m_H65w3L-_VrA");
        checksums.put("DenseNet121", 3218375330L);
        checksums.put("DenseNet169", 618879071L);
        checksums.put("DenseNet201", 2157344866L);

        // ############
        // EfficientNet
        // ############
        locations.put("EfficientNetB0", "https://drive.google.com/uc?export=download&id=1osvsogvbEKyT0VIZ9JtZhfgLQKCiPU8U");
        locations.put("EfficientNetB1", "https://drive.google.com/uc?export=download&id=1VpMDTrSYkUs-6-Ug6FCrBX_G2WVbXoNp");
        locations.put("EfficientNetB2", "https://drive.google.com/uc?export=download&id=1omrSGk5x2_lTfPSGjfIzpYlr8cU4WPbi");
        locations.put("EfficientNetB3", "https://drive.google.com/uc?export=download&id=1e0gaLQ6vsCqe4mMHYDaNkd5jafXfS7HZ");
        locations.put("EfficientNetB4", "https://drive.google.com/uc?export=download&id=1NDqrSBZ-p6Ct70ZwbBsZOL3YAAN-0ypz");
        locations.put("EfficientNetB5", "https://drive.google.com/uc?export=download&id=1p6Hd5GNZn0R-xf4LXumFIYYAL2Jbu69y");
        locations.put("EfficientNetB6", "https://drive.google.com/uc?export=download&id=1AoZunscMwqq7UhpirxfMyRMeN10NYaG6");
        locations.put("EfficientNetB7", "https://drive.google.com/uc?export=download&id=1IvejTyjhXbPWDuPW5HyogwD2dv0e2Nvg");
        checksums.put("EfficientNetB0", 3990389537L);
        checksums.put("EfficientNetB1", 2300415769L);
        checksums.put("EfficientNetB2", 2993104358L);
        checksums.put("EfficientNetB3", 1778527599L);
        checksums.put("EfficientNetB4", 3090591408L);
        checksums.put("EfficientNetB5", 620292868L);
        checksums.put("EfficientNetB6", 3557280118L);
        checksums.put("EfficientNetB7", 1866748858L);

        // #################
        // InceptionResNetV2
        // #################
        locations.put("InceptionResNetV2", "https://drive.google.com/uc?export=download&id=1pXfR8UQnDjvEE5QVg6RDAkwtQ9nHH0cx");
        checksums.put("InceptionResNetV2", 2782798852L);

        // ###########
        // InceptionV3
        // ###########
        locations.put("InceptionV3", "https://drive.google.com/uc?export=download&id=19BxOrIFjUMyu_IQv6AKAufTxdnPIXrQ5");
        checksums.put("InceptionV3", 2734012442L);

        // #########
        // MobileNet
        // #########
        locations.put("MobileNet", "https://drive.google.com/uc?export=download&id=1knZxjDsF6uWSMJNGRrY8eAcRZZyqB8Ar");
        locations.put("MobileNetV2", "https://drive.google.com/uc?export=download&id=14_YGMGndsbTpcFdmlWw84dwSqHHZIo4a");
        checksums.put("MobileNet", 1615806516L);
        checksums.put("MobileNetV2", 1015291493L);

        // ######
        // NASNet
        // ######
        locations.put("NASNetMobile", "https://drive.google.com/uc?export=download&id=1bDkDO-UUvYxZENg5CTwJPAAs1jaGly6z");
        locations.put("NASNetLarge", "https://drive.google.com/uc?export=download&id=1_lDpE9U9Y7y4_Tqjc-RvACT9Gf372j_z");
        checksums.put("NASNetMobile", 2888382958L);
        checksums.put("NASNetLarge", 914859199L);

        // ######
        // ResNet
        // ######
        locations.put("ResNet50", "https://drive.google.com/uc?export=download&id=1VkHHHgWtH0rZf00kajhd46KB08L50yng");
        locations.put("ResNet50V2", "https://drive.google.com/uc?export=download&id=112HAXAg_FXk6mfwsWrnJ4d1lzZ3aFhgk");
        locations.put("ResNet101", "https://drive.google.com/uc?export=download&id=1tc-EZvCvddYGDB0RyJVhcHscvWy5dabE");
        locations.put("ResNet101V2", "https://drive.google.com/uc?export=download&id=1amU0nCXEI79tRF1qH1MTKTiOUPJkHzaB");
        locations.put("ResNet152", "https://drive.google.com/uc?export=download&id=1j48X1YQBMpozT6LQz0PI_VyCjKmA60DJ");
        locations.put("ResNet152V2", "https://drive.google.com/uc?export=download&id=1hF42bWVH0qGaTannNJXbMO_NON_rOvBN");
        checksums.put("ResNet50", 3696145880L);
        checksums.put("ResNet50V2", 2441552962L);
        checksums.put("ResNet101", 1250607721L);
        checksums.put("ResNet101V2", 1889483812L);
        checksums.put("ResNet152", 503735698L);
        checksums.put("ResNet152V2", 4255384576L);
        Locations = Collections.unmodifiableMap(locations);
        Checksums = Collections.unmodifiableMap(checksums);
    }
}
