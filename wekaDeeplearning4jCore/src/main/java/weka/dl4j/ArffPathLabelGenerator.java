package weka.dl4j;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class ArffPathLabelGenerator implements PathLabelGenerator {

    /**
     * Mapping from paths to labels
     */
    private Map<String, String> fileLabelMap;

    /**
     * Paths of the input files
     */
    private List<String> paths;

    /**
     * Labels of the input files
     */
    private List<String> labels;

    /**
     * Basepath where the files reside
     */
    private String basePath;

    /**
     * Default constructor which sets the metaData
     *
     * @param metaData Meta data with mapping: filename to label
     * @param path Directory path
     */
    public ArffPathLabelGenerator(Instances metaData, String path) {

        // If this path is absolute set it as basepath
        if (new File(path).isAbsolute()){
            this.basePath = path;
        } else {
            String currentPath = Paths.get(System.getProperty("user.dir")).toString();
            this.basePath = Paths.get(currentPath, path).toString();
        }

        // Fill mapping from image path to
        fileLabelMap = new HashMap<>();
        paths = new ArrayList<>();
        labels = new ArrayList<>();
        for (Instance inst : metaData) {
            String fileName = inst.stringValue(0);
            String label = inst.stringValue(1);
            String absPath = Paths.get(this.basePath, fileName).toString();
            paths.add(absPath);
            labels.add(label);
            fileLabelMap.put(absPath, label);
        }
    }

    /**
     * Select the label based on the path in the metadata
     * @param path Input path to the image
     * @return Label of the image at the given path
     */
    @Override
    public Writable getLabelForPath(String path) {
        String absPath = URI.create(path).getRawPath();
        final String label = fileLabelMap.get(absPath);
        return new Text(label);
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(uri.toString());
    }

    /**
     * Get all paths as uris
     * @return Collection of uri paths
     */
    public Collection<URI> getPathURIs(){
        return paths.stream()
                .map(s -> Paths.get(s).toUri())
                .collect(Collectors.toList());
    }

}
