package weka.dl4j.scripts;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * This class loads in a folder of Keras files, and one by one converts them
 * into the native DL4J format (.zip). This is safer to work with in DL4J than
 * importing from Keras files every time, and is fine to do in this case because
 * WDL4J defines a fixed set of models - this process only needs to be done once.
 */
public class KerasModelConverter {

    // Default location where Keras models are saved
    private static String modelFolderPath = "";

    private static void saveH5File(File modelFile, File outputFolder) {
        try {
            ComputationGraph kerasModel = KerasModelImport.importKerasModelAndWeights(modelFile.getAbsolutePath());
            kerasModel.feedForward(Nd4j.zeros(1, 3, 224, 224), false);
            // e.g. ResNet50.h5 -> KerasResNet50.zip
            String modelName = "Keras" + modelFile.getName().replace(".h5", ".zip");
            String newZip = Paths.get(outputFolder.getPath(), modelName).toString();
            kerasModel.save(new File(newZip));
            System.out.println("Saved file " + newZip);
        } catch (Exception e) {
            System.err.println("\n\nCouldn't save " + modelFile.getName());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        System.err.println("WARNING: This conversion script should be run with DL4J 1.0.0-beta6, " +
                "any other version may result in non-working model files");

        if (args.length == 1) {
            modelFolderPath = args[0];
        }

        File modelFolder = new File(modelFolderPath);
        File outputFolder = new File(Paths.get(modelFolder.getParent(), "dl4j_format").toString());
        if (outputFolder.mkdir())
            System.out.println("Created DL4J format folder at " + outputFolder.getPath());

        File[] modelFiles = modelFolder.listFiles();
        assert modelFiles != null;
        Arrays.sort(modelFiles);
        for (File fileEntry : modelFiles) {
            if (fileEntry.getPath().endsWith(".h5")) {
                saveH5File(fileEntry, outputFolder);
            }
        }
    }
}
