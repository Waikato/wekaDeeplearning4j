package weka.dl4j.zoo.keras;

import java.util.Collections;
import java.util.Dictionary;
import java.util.HashMap;
import java.util.Map;

public class KerasConstants {

    public static Map<String, String> Locations;
    public static Map<String, Long> Checksums;

    static {
        Map<String, String> locMap = new HashMap<>();
        locMap.put("DenseNet121", "https://drive.google.com/uc?export=download&id=1skHkrjXhCv10-ASZelLaF98lAxOB9c1-");
        locMap.put("DenseNet169", "https://drive.google.com/uc?export=download&id=1n6E--TOXBmsLoROzYHI7KJYpSe4BJFS5");
        locMap.put("DenseNet201", "https://drive.google.com/uc?export=download&id=10Ki6JCHmB27LJw1nKs4m_H65w3L-_VrA");
        Locations = Collections.unmodifiableMap(locMap);

        Map<String, Long> checkMap = new HashMap<>();
        checkMap.put("DenseNet121", 3218375330L);
        checkMap.put("DenseNet169", 618879071L);
        checkMap.put("DenseNet201", 2157344866L);
        Checksums = Collections.unmodifiableMap(checkMap);
    }
}
