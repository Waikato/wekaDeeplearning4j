package weka.gui.explorer;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FilenameUtils;
import weka.core.progress.ProgressManager;
import weka.dl4j.inference.Dl4jCNNExplorer;
import weka.dl4j.inference.PredictionClass;
import weka.dl4j.interpretability.AbstractCNNSaliencyMapWrapper;
import weka.gui.ExtensionFileFilter;
import weka.gui.WekaFileChooser;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Pattern;

@Log4j2
public class SaliencyMapWindow extends JPanel {



    Dl4jCNNExplorer processedExplorer;
    /**
     * UI Elements
     */
    JFrame saliencyMapWindow = new JFrame("WekaDeeplearning4j - Saliency Map Viewer");
    JLabel targetClassIDLabel = new JLabel("Target Class ID:");
    JTextField targetClassIDInput = new JTextField();
    JLabel classNameLabel = new JLabel("  Class Name:");
    JTextField classNameInput = new JTextField();
    JButton patternButton = new JButton("Pattern");
    /** The current regular expression. */
    String m_PatternRegEx = "";
    JButton generateButton = new JButton("Generate");
    JLabel saliencyImageLabel = new JLabel();
    Image saliencyImage;
    JCheckBox normalizeHeatmapCheckbox = new JCheckBox("Normalize heatmap");
    JButton saveHeatmapButton = new JButton("Save...");
    private static final String DEFAULT_SALIENCY_IMAGE_PATH = "src/main/resources/placeholderSaliencyMap.png";

    protected FileFilter m_ImageFilter = new ExtensionFileFilter(ExplorerDl4jInference.IMAGE_FILE_EXTENSIONS, "Image files");

    /** The file chooser for selecting model files. */
    protected WekaFileChooser m_FileChooser = new WekaFileChooser(new File(System.getProperty("user.dir")));

    public SaliencyMapWindow() {
        setup();
    }

    private void setup() {
        // Saliency Map Window
        patternButton.addActionListener(e -> openPatternDialog());
        generateButton.addActionListener(e -> generateSaliencyMap());
        saveHeatmapButton.addActionListener(e -> saveHeatmap());
        normalizeHeatmapCheckbox.addActionListener(e -> generateSaliencyMap());

        // Setup the button listeners
        targetClassIDInput.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                updateClassNameInput();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                updateClassNameInput();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                updateClassNameInput();
            }
        });

        // Define the UI elements
        targetClassIDInput.setColumns(5);
        targetClassIDInput.setToolTipText("-1 to use max probability class");
        classNameInput.setColumns(40);
        classNameInput.setEditable(false);

        // Panel to define the layout. We are using GridBagLayout
        JPanel mainPanel = new JPanel();
        GridBagLayout gbL = new GridBagLayout();
        mainPanel.setLayout(gbL);
        mainPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Saliency Map Viewer"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5)));

        // Setup top row in saliency window - labels and text fields
        JPanel topRow = new JPanel(new FlowLayout(FlowLayout.LEFT));
        topRow.add(targetClassIDLabel);
        topRow.add(targetClassIDInput);
        topRow.add(classNameLabel);
        topRow.add(classNameInput);
        topRow.add(patternButton);
        GridBagConstraints gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridx = 0;
        gbC.gridy = 0;
        gbL.setConstraints(topRow, gbC);
        mainPanel.add(topRow);

        // Setup second row - Normalize checkbox and Generate button
        JPanel secondRow = new JPanel(new GridLayout(1, 3, 30, 5));
        secondRow.add(normalizeHeatmapCheckbox);
        secondRow.add(generateButton);
        secondRow.add(saveHeatmapButton);
        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridx = 0;
        gbC.gridy = 1;
        gbC.insets = new Insets(5, 0, 20, 0);
        gbL.setConstraints(secondRow, gbC);
        mainPanel.add(secondRow);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.CENTER;
        gbC.gridx = 0;
        gbC.gridy = 2;
        gbL.setConstraints(saliencyImageLabel, gbC);
        mainPanel.add(saliencyImageLabel);

        // Add panel to frame
        saliencyMapWindow.add(mainPanel);
    }

    public void open(Dl4jCNNExplorer explorer) {
        this.processedExplorer = explorer;

        // Reset the image
        try {
            setSaliencyImage(ImageIO.read(new File(DEFAULT_SALIENCY_IMAGE_PATH)));
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        // Set the default class ID in the window
        setTargetClass(getDefaultClassID());
        normalizeHeatmapCheckbox.setSelected(true);

        saliencyMapWindow.pack();
        saliencyMapWindow.setLocationRelativeTo(null);
        saliencyMapWindow.setVisible(true);
    }

    private void generateSaliencyMap() {
        SwingWorker<Image, Void> worker = new SwingWorker<Image, Void>() {
            @Override
            protected Image doInBackground() {
                ProgressManager manager = new ProgressManager(-1, "Generating saliency map...");
                manager.start();
                int targetClassID = Integer.parseInt(targetClassIDInput.getText());
                boolean normalize = normalizeHeatmapCheckbox.isSelected();
                log.info("Generating for class = " + targetClassID);

                AbstractCNNSaliencyMapWrapper wrapper = processedExplorer.getSaliencyMapGenerator();
                wrapper.setTargetClassID(targetClassID);
                wrapper.setNormalizeHeatmap(normalize);

                processedExplorer.setSaliencyMapGenerator(wrapper);
                Image outputMap = processedExplorer.generateOutmapToImage();

                manager.finish();

                return outputMap;
            }

            @SneakyThrows
            @Override
            protected void done() {
                super.done();
                setSaliencyImage(get());

                // Pack
                saliencyMapWindow.pack();
            }
        };
        worker.execute();
    }

    private void setTargetClass(int id) {
        targetClassIDInput.setText("" + id);
    }

    private int getDefaultClassID() {
        if (processedExplorer != null)
            return processedExplorer.getCurrentPredictions().getTopPrediction().getClassID();
        else
            return -1;
    }

    private void setSaliencyImage(Image image) {
        saliencyImage = image;
        ImageIcon icon = new ImageIcon(saliencyImage);
        saliencyImageLabel.setIcon(icon);
        saliencyImageLabel.invalidate();
    }

    private void openPatternDialog() {
        String pattern = JOptionPane.showInputDialog(patternButton.getParent(),
                "Enter a Perl regular expression", m_PatternRegEx);
        if (pattern != null) {
            try {
                Pattern.compile(pattern);
                m_PatternRegEx = pattern;
                ArrayList<PredictionClass> matchingClasses = getMatchingClasses(pattern);

                if (matchingClasses.isEmpty()) {
                    JOptionPane.showMessageDialog(null, "No classes matched that regex pattern");
                    return;
                } else if (matchingClasses.size() == 1) {
                    setTargetClass(matchingClasses.get(0).getID());
                } else {
                    PredictionClass selectedClass = selectOneOfNClasses(matchingClasses);
                    setTargetClass(selectedClass.getID());
                }

            } catch (Exception ex) {
                JOptionPane.showMessageDialog(patternButton.getParent(), "'" + pattern
                                + "' is not a valid Perl regular expression!\n" + "Error: " + ex,
                        "Error in Pattern...", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private PredictionClass selectOneOfNClasses(ArrayList<PredictionClass> matchingClasses) {
        return (PredictionClass) JOptionPane.showInputDialog(
                null,
                "The pattern matched multiple classes, please select one",
                "Select a class",
                JOptionPane.QUESTION_MESSAGE,
                null,
                matchingClasses.toArray(), // Array of choices
                matchingClasses.get(0));
    }

    private ArrayList<PredictionClass> getMatchingClasses(String pattern) {
        ArrayList<PredictionClass> result = new ArrayList<>();
        String[] classMap;
        try {
            classMap = processedExplorer.getModelOutputDecoder().getClasses();
        } catch (Exception ex) {
            classMap = new String[]{};
        } // Change to ClassificationClass
        for (int i = 0; i < classMap.length; i++) {
            String tmpClass = classMap[i];
            if (Pattern.matches(pattern, tmpClass)) {
                result.add(new PredictionClass(i, tmpClass));
            }
        }
        return result;
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

    private String getClassName(int classID) {
        String[] classMap;
        try {
            classMap = processedExplorer.getModelOutputDecoder().getClasses();
        } catch (Exception ex) {
            classMap = new String[]{};
        }
        try {
            return classMap[classID];
        } catch (IndexOutOfBoundsException ex) {
            return null;
        }
    }

    private void updateClassNameInput() {
        String targetClassIDText = targetClassIDInput.getText();
        if (targetClassIDText.isEmpty()) {
            return;
        }
        int classID;
        try {
            classID = Integer.parseInt(targetClassIDText);
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(null,
                    "Error: Please enter a valid integer value", "Error Message",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        String newClassName = getClassName(classID);
        classNameInput.setText(newClassName);
    }
}
