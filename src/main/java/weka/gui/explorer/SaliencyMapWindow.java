package weka.gui.explorer;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FilenameUtils;
import weka.core.WekaException;
import weka.core.progress.ProgressManager;
import weka.dl4j.ResourceResolver;
import weka.dl4j.inference.Dl4jCNNExplorer;
import weka.dl4j.interpretability.AbstractCNNSaliencyMapWrapper;
import weka.gui.ExtensionFileFilter;
import weka.gui.WekaFileChooser;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

@Log4j2
public class SaliencyMapWindow extends JPanel {

    public static class SaliencyMapGBC extends GridBagConstraints {
        public SaliencyMapGBC() {
            this(5);
        }

        public SaliencyMapGBC(int insetSize) {
            this.insets = new Insets(insetSize, insetSize, insetSize, insetSize);
        }
    }

    Dl4jCNNExplorer processedExplorer;
    /**
     * UI Elements
     */
    JFrame thisWindow = new JFrame("WekaDeeplearning4j - Saliency Map Viewer");

    JLabel saliencyImageLabel;
    ImageIcon icon;
    Image saliencyImage;
    JCheckBox normalizeHeatmapCheckbox = new JCheckBox("Normalize heatmap");
    JButton addClassButton = new JButton("Add Class");
    JButton removeClassButton = new JButton("Remove Class");
    JButton generateButton = new JButton("Generate");
    JButton saveHeatmapButton = new JButton("Save...");
    JScrollPane scrollPane;
    JPanel buttonPanel;
    JPanel controlPanel;
    JSplitPane splitPane;

    private String DEFAULT_SALIENCY_IMAGE_PATH;

    protected FileFilter m_ImageFilter = new ExtensionFileFilter(ExplorerDl4jInference.IMAGE_FILE_EXTENSIONS, "Image files");

    /** The file chooser for saving the image. */
    protected WekaFileChooser m_FileChooser = new WekaFileChooser(new File(System.getProperty("user.dir")));

    ArrayList<ClassSelector> classSelectors = new ArrayList<>();

    int buttonsRow = 0;
    int targetClassRow = 1;
    int imageRow = 20;

    public SaliencyMapWindow() {
        try {
            DEFAULT_SALIENCY_IMAGE_PATH = new ResourceResolver().GetResolvedPath("placeholderSaliencyMap.png");
        } catch (WekaException ex) {
            ex.printStackTrace();
        }

        oneTimeSetup();
    }

    private void addClassSelector() {
        if (classSelectors.size() == 5) {
            // Limit the size to 10
            return;
        }
        ClassSelector classSelector = new ClassSelector(controlPanel, getCurrentClassMap(), getDefaultClassID(), targetClassRow + classSelectors.size());
        classSelectors.add(classSelector);
        packWindow();
    }

    private void clearClassSelectors() {
        while (classSelectors.size() > 0) {
            removeClassSelector();
        }
    }

    private void removeClassSelector() {
        if (classSelectors.size() == 0) {
            // Don't go below one class selector
            return;
        }
        ClassSelector lastClassSelector = classSelectors.get(classSelectors.size() - 1);
        lastClassSelector.removeFromParent();
        classSelectors.remove(lastClassSelector);
        packWindow();
    }

    private void packWindow() {
//        var originalDimension = thisWindow.getSize();
        thisWindow.pack();
        splitPane.setDividerLocation(splitPane.getMinimumDividerLocation());
//        thisWindow.setSize(originalDimension);
    }

    private void setupButtonListeners() {
        generateButton.addActionListener(e -> generateSaliencyMap());
        saveHeatmapButton.addActionListener(e -> saveHeatmap());
        normalizeHeatmapCheckbox.addActionListener(e -> generateSaliencyMap());
        addClassButton.addActionListener(e -> addClassSelector());
        removeClassButton.addActionListener(e -> removeClassSelector());

        // Set the default class ID in the window
        normalizeHeatmapCheckbox.setSelected(true);
    }

    private void addControlButtons() {
        GridBagConstraints gbc = new SaliencyMapGBC();
        gbc.gridy = buttonsRow;

        gbc.gridx = 0;
        gbc.gridwidth = 5;
        buttonPanel = new JPanel(new GridLayout(1, 5, 30, 5));
        buttonPanel.add(addClassButton);
        buttonPanel.add(removeClassButton);
        buttonPanel.add(normalizeHeatmapCheckbox);
        buttonPanel.add(generateButton);
        buttonPanel.add(saveHeatmapButton);
        controlPanel.add(buttonPanel, gbc);
    }

    private void addScrollableImage() {
        GridBagConstraints gbc = new SaliencyMapGBC();
        gbc.gridx = 0;
        gbc.gridy = imageRow;
        gbc.weighty = 1;
        gbc.fill = GridBagConstraints.BOTH;
        gbc.gridwidth = 5;
        gbc.gridheight = 1;
        saliencyImageLabel = new JLabel(new ImageIcon(DEFAULT_SALIENCY_IMAGE_PATH));
        scrollPane = new JScrollPane(saliencyImageLabel);
        scrollPane.getVerticalScrollBar().setUnitIncrement(20);
        scrollPane.getHorizontalScrollBar().setUnitIncrement(20);
    }

    private void oneTimeSetup() {
        setupButtonListeners();

        setLayout(new BorderLayout());
        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Saliency Map Viewer"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5)));

        controlPanel = new JPanel(new GridBagLayout());

        addControlButtons();

        addScrollableImage();

        splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT, controlPanel, scrollPane);
        splitPane.setOneTouchExpandable(true);

        add(splitPane, BorderLayout.CENTER);
    }

    public String[] getCurrentClassMap() {
        if (processedExplorer == null) {
            return new String[0];
        } else {
            return processedExplorer.getModelOutputDecoder().getClasses();
        }
    }

    public void open(Dl4jCNNExplorer explorer) {
        this.processedExplorer = explorer;

        // Start with a fresh class selector
        clearClassSelectors();
        addClassSelector();

        thisWindow.add(this);
        packWindow();
        thisWindow.setLocationRelativeTo(null);
        thisWindow.setVisible(true);

        generateSaliencyMap();
    }

    public int getDefaultClassID() {
        if (processedExplorer != null)
            return processedExplorer.getCurrentPredictions().getTopPrediction().getClassID();
        else
            return -1;
    }

    private int[] getTargetClassIDs() {
        return classSelectors.stream().mapToInt(ClassSelector::getTargetClass).toArray();
    }

    private void validateClassID() {
        Integer[] targetClasses = classSelectors.stream().map(ClassSelector::getTargetClass).toArray(Integer[]::new);
        log.info("Generating for classes = " + Arrays.toString(targetClasses));

        for (Integer targetClass : targetClasses) {
            if (targetClass < 0 || targetClass >= getCurrentClassMap().length) {
                throw new IllegalArgumentException("Invalid class ID(s) supplied: " + Arrays.toString(targetClasses));
            }
        }
    }

    private void generateSaliencyMap() {
        validateClassID();
        SwingWorker<Image, Void> worker = new SwingWorker<Image, Void>() {
            @Override
            protected Image doInBackground() {
                ProgressManager manager = new ProgressManager(-1, "Generating saliency map...");
                manager.start();
                boolean normalize = normalizeHeatmapCheckbox.isSelected();

                AbstractCNNSaliencyMapWrapper wrapper = processedExplorer.getSaliencyMapWrapper();
                wrapper.setTargetClassIDsAsInt(getTargetClassIDs());
                wrapper.setNormalizeHeatmap(normalize);

                processedExplorer.setSaliencyMapWrapper(wrapper);
                Image outputMap = processedExplorer.generateOutputMap();
                manager.finish();

                return outputMap;
            }

            @SneakyThrows
            @Override
            protected void done() {
                super.done();
                setSaliencyImage(get());

                // Pack
                packWindow();
            }
        };
        worker.execute();
    }

    private void setSaliencyImage(Image image) {
        if (image == null) {
            try {
                image = ImageIO.read(new File(DEFAULT_SALIENCY_IMAGE_PATH));
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        saliencyImage = image;
        icon = new ImageIcon(saliencyImage);
        saliencyImageLabel.setIcon(icon);
        saliencyImageLabel.invalidate();
    }

    private void saveHeatmap() {
        // Prompt the user for a place to save the image to
        m_FileChooser.setFileFilter(m_ImageFilter);

        int returnCode = m_FileChooser.showSaveDialog(this);

        if (returnCode == 1) {
            log.error("User did not select a new image");
            return;
        }
        // Get the image
        File outputFile = m_FileChooser.getSelectedFile();

        String extension = FilenameUtils.getExtension(outputFile.getName());

        try {
            ImageIO.write((RenderedImage) saliencyImage, extension, outputFile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
}
