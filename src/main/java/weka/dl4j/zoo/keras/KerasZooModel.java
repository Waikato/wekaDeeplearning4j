package weka.dl4j.zoo.keras;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

@Log4j2
public abstract class KerasZooModel extends ZooModel {

    protected int[] inputShape;

    public abstract String modelFamily();

    public abstract String modelPrettyName();

    public abstract void setVariation(Enum variation);

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    @Override
    public ComputationGraph init() {
        // We initialize the model in initPretrained()
        return null;
    }

    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET) {
            return KerasConstants.Locations.get(modelPrettyName());
        } else {
            return null;
        }
    }

    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET) {
            return KerasConstants.Checksums.get(modelPrettyName());
        } else {
            return 0L;
        }
    }

    @Override
    public ComputationGraph initPretrained(PretrainedType pretrainedType) throws IOException {
        String remoteUrl = pretrainedUrl(pretrainedType);
        if (remoteUrl == null)
            throw new UnsupportedOperationException(
                    "Pretrained " + pretrainedType + " weights are not available for this model.");

        String localFilename = modelPrettyName() + ".h5";

        File rootCacheDir = DL4JResources.getDirectory(ResourceType.ZOO_MODEL, modelFamily());
        File cachedFile = new File(rootCacheDir, localFilename);

        if (!cachedFile.exists()) {
            log.info("Downloading model to " + cachedFile.toString());
            FileUtils.copyURLToFile(new URL(remoteUrl), cachedFile);
        } else {
            log.info("Using cached model at " + cachedFile.toString());
        }

        long expectedChecksum = pretrainedChecksum(pretrainedType);
        if (expectedChecksum != 0L) {
            log.info("Verifying download...");
            Checksum adler = new Adler32();
            FileUtils.checksum(cachedFile, adler);
            long localChecksum = adler.getValue();
            log.info("Checksum local is " + localChecksum + ", expecting " + expectedChecksum);

            if (expectedChecksum != localChecksum) {
                log.error("Checksums do not match. Cleaning up files and failing...");
                cachedFile.delete();
                throw new IllegalStateException(
                        String.format("Pretrained model file for model %s failed checksum.", this.modelPrettyName()));
            }
        }

        try {
            return KerasModelImport.importKerasModelAndWeights(cachedFile.getPath());
        } catch (Exception ex) {
            System.err.println("Failed to load model");
            ex.printStackTrace();
            return null;
        }
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

    @Override
    public ModelMetaData metaData() {
        return null;
    }
}
