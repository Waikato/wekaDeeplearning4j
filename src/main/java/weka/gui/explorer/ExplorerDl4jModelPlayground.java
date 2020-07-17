package weka.gui.explorer;

import lombok.SneakyThrows;
import weka.core.*;

import weka.dl4j.playground.Dl4jCNNExplorer;
import weka.dl4j.zoo.AbstractZooModel;
import weka.dl4j.zoo.Dl4jResNet50;
import weka.gui.*;
import weka.gui.explorer.Explorer.ExplorerPanel;
import weka.gui.explorer.Explorer.LogHandler;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.event.*;
import java.io.File;

public class ExplorerDl4jModelPlayground extends JPanel implements ExplorerPanel, LogHandler {

    /** the parent frame */
    protected Explorer m_Explorer = null;

//    /** sends notifications when the set of working instances gets changed*/
//    protected PropertyChangeSupport m_Support = new PropertyChangeSupport(this);

    protected Instances m_Instances = null;

    protected Logger m_Logger = new SysErrLog();

    protected String m_currentlyDisplayedImage = "";

    /**
     * UI Components
     */

    /** Lets the user configure the classifier. */
    protected GenericObjectEditor m_CNNExplorerEditor = new GenericObjectEditor();

    /** The panel showing the current classifier selection. */
    protected PropertyPanel m_playgroundEditorPanel = new PropertyPanel(m_CNNExplorerEditor);

    /** The filename extension that should be used for model files. */
    public static String MODEL_FILE_EXTENSION = ".model";

    /** The filename extension that should be used for PMML xml files. */
    public static String PMML_FILE_EXTENSION = ".xml";

    public static String[] IMAGE_FILE_EXTENSIONS = new String[] {".jpg", ".jpeg", ".png"};

    /** The output area for classification results. */
    protected JTextArea m_OutText = new JTextArea(10, 40);

    /** A panel controlling results viewing. */
    protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);

    /** Click to set test mode to test on training data. */
    protected JRadioButton m_modelFileRadio = new JRadioButton("Use Model File");

    /** Click to set test mode to a user-specified test set. */
    protected JRadioButton m_ZooModelRadio = new JRadioButton("Zoo Model");

    /** Button for further output/visualize options. */
    JButton m_OpenImageButton = new JButton("Open Image...");

    /** The button used to open a separate test dataset. */
    protected JButton m_setModelFileBut = new JButton("Set...");

    /** The button used to open a separate test dataset. */
    protected JButton m_setZooModelBut = new JButton("Set...");

    /**
     * Alters the enabled/disabled status of elements associated with each radio
     * button.
     */
    ActionListener m_RadioListener = e -> updateRadioLinks();


    /** Click to start running the classifier. */
    protected JButton m_predictButton = new JButton("Predict");

    /** A thread that classification runs in. */
    protected Thread m_RunThread;

    /** Filter to ensure only model files are selected. */
    protected FileFilter m_ModelFilter = new ExtensionFileFilter(
            MODEL_FILE_EXTENSION, "Model object files");

    protected FileFilter m_PMMLModelFilter = new ExtensionFileFilter(
            PMML_FILE_EXTENSION, "PMML model files");

    protected FileFilter m_ImageFilter = new ExtensionFileFilter(
            IMAGE_FILE_EXTENSIONS, "Image files");

    /** The file chooser for selecting model files. */
    protected WekaFileChooser m_FileChooser = new WekaFileChooser(new File(
            System.getProperty("user.dir")));

    JLabel imageLabel;

    JPanel imagePanel;

    protected AbstractZooModel zooModel = new Dl4jResNet50();

    protected Dl4jCNNExplorer imageModelPlayground;

    /* Register the property editors we need */
    static {
        GenericObjectEditor.registerEditors();
    }

    /**
     * Sets the Explorer to use as parent frame (used for sending notifications
     * about changes in the data)
     *
     * @param parent the parent frame
     */
    @Override
    public void setExplorer(Explorer parent) {
        m_Explorer = parent;
    }

    /**
     * returns the parent Explorer frame
     *
     * @return the parent
     */
    @Override
    public Explorer getExplorer() {
        return m_Explorer;
    }

    /**
     * Tells the panel to use a new set of instances.
     *
     * @param inst a set of Instances
     */
    @Override
    public void setInstances(Instances inst) {
        m_Instances = inst;
    }

    /**
     * Returns the title for the tab in the Explorer
     *
     * @return the title of this tab
     */
    @Override
    public String getTabTitle() {
        return "Dl4j Model Explorer";
    }

    /**
     * Returns the tooltip for the tab in the Explorer
     *
     * @return the tooltip of this tab
     */
    @Override
    public String getTabTitleToolTip() {
        return "A playground for trying different trained classification models on individual images.";
    }

    /**
     * Sets the Logger to receive informational messages
     *
     * @param newLog the Logger that will now get info messages
     */
    @Override
    public void setLog(Logger newLog) {
        m_Logger = newLog;
    }

    public ExplorerDl4jModelPlayground() {
        super();

        initGUI();
    }

    protected void initGUI() {
        setupPlaygroundEditor();

        setupOutputText();

        JPanel historyPanel = setupHistoryPanel();

        setupToolTipText();

        setupFileChooser();

        setupButtonListeners();

        JPanel optionsPanel = setupPlaygroundOptions();

        JPanel buttons = setupStartButton();

        JPanel modelOutput = setupOutputPanel();

        JPanel imagePanel = setupImagePanel();

        setupMainLayout(optionsPanel, buttons, historyPanel, modelOutput, imagePanel);

        setDefaultRadioButton();
    }

    private void setupPlaygroundEditor() {
        m_CNNExplorerEditor.setClassType(Dl4jCNNExplorer.class);
        m_CNNExplorerEditor.setValue(new Dl4jCNNExplorer());
    }

    private void setupOutputText() {
        // Connect / configure the components
        m_OutText.setEditable(false);
        m_OutText.setFont(new Font("Monospaced", Font.PLAIN, 12));
        m_OutText.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        m_OutText.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if ((e.getModifiers() & InputEvent.BUTTON1_MASK) != InputEvent.BUTTON1_MASK) {
                    m_OutText.selectAll();
                }
            }
        });
    }

    private JPanel setupHistoryPanel() {
        JPanel historyPanel = new JPanel(new BorderLayout());
        historyPanel.setBorder(BorderFactory
                .createTitledBorder("Result list (right-click for options)"));
        historyPanel.add(m_History, BorderLayout.CENTER);
        m_History.setHandleRightClicks(false);

        return historyPanel;
    }

    private void setupToolTipText() {
        m_modelFileRadio.setToolTipText("Test on the same set that the classifier"
                + " is trained on");

        m_ZooModelRadio.setToolTipText("Test on a user-specified dataset");
        m_predictButton.setToolTipText("Starts the classification");
    }

    private void setupFileChooser() {
//        m_FileChooser.setFileFilter(m_ModelFilter);
        m_FileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
    }

    private void setupButtonListeners() {
        ButtonGroup bg = new ButtonGroup();
        bg.add(m_modelFileRadio);
        bg.add(m_ZooModelRadio);

        m_modelFileRadio.addActionListener(m_RadioListener);
        m_ZooModelRadio.addActionListener(m_RadioListener);

        m_setZooModelBut.addActionListener(e -> {
//                System.out.println("Opening Zoo model prompt");
//                setTestSet();
        });

        m_setModelFileBut.addActionListener(actionEvent -> System.out.println("Opening model file prompt..."));

        m_OpenImageButton.addActionListener(actionEvent -> openNewImage());

        m_predictButton.setEnabled(false);
        m_predictButton.addActionListener(e -> {
            predict();
        });
    }

    private JPanel setupPlaygroundOptions() {

        JPanel optionsPanel = new JPanel();
        GridBagLayout gbL = new GridBagLayout();
        optionsPanel.setLayout(gbL);
        optionsPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Playground options"),
                BorderFactory.createEmptyBorder(0, 5, 5, 5)));

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.WEST;
        gbC.gridy = 0;
        gbC.gridx = 0;
        gbL.setConstraints(m_modelFileRadio, gbC);
        optionsPanel.add(m_modelFileRadio);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.EAST;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 0;
        gbC.gridx = 1;
        gbC.gridwidth = 2;
        gbC.insets = new Insets(2, 10, 2, 0);
        gbL.setConstraints(m_setModelFileBut, gbC);
        optionsPanel.add(m_setModelFileBut);


        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.WEST;
        gbC.gridy = 1;
        gbC.gridx = 0;
        gbL.setConstraints(m_ZooModelRadio, gbC);
        optionsPanel.add(m_ZooModelRadio);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.EAST;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 1;
        gbC.gridx = 1;
        gbC.gridwidth = 2;
        gbC.insets = new Insets(2, 10, 2, 0);
        gbL.setConstraints(m_setZooModelBut, gbC);
        optionsPanel.add(m_setZooModelBut);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.WEST;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 4;
        gbC.gridx = 0;
        gbC.weightx = 100;
        gbC.gridwidth = 3;
        gbC.insets = new Insets(3, 0, 1, 0);
        gbL.setConstraints(m_OpenImageButton, gbC);
        optionsPanel.add(m_OpenImageButton);

        return optionsPanel;
    }

    private JPanel setupStartButton() {
        JPanel buttons = new JPanel();
        buttons.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
        buttons.setLayout(new FlowLayout(FlowLayout.CENTER));
        buttons.add(m_predictButton);

        return buttons;
    }

    private JPanel setupOutputPanel() {
        JPanel outputPanel = new JPanel();
        outputPanel.setBorder(BorderFactory.createTitledBorder("Model output"));
        outputPanel.setLayout(new BorderLayout());
        final JScrollPane js = new JScrollPane(m_OutText);
        outputPanel.add(js, BorderLayout.CENTER);
        js.getViewport().addChangeListener(new ChangeListener() {
            private int lastHeight;

            @Override
            public void stateChanged(ChangeEvent e) {
                JViewport vp = (JViewport) e.getSource();
                int h = vp.getViewSize().height;
                if (h != lastHeight) { // i.e. an addition not just a user scrolling
                    lastHeight = h;
                    int x = h - vp.getExtentSize().height;
                    System.out.println(x);
                    vp.setViewPosition(new Point(0, x));
                }
            }
        });
        return outputPanel;
    }

    private void setDefaultRadioButton() {
        m_ZooModelRadio.setSelected(true);
        updateRadioLinks();
    }

    private void setupMainLayout(JPanel optionsPanel, JPanel buttons, JPanel historyPanel, JPanel outputPanel, JPanel imagePanel) {
        JPanel mainPanel = new JPanel();
        GridBagLayout mainLayout = new GridBagLayout();
        mainPanel.setLayout(mainLayout);

        // Layout the GUI
        JPanel topPanel = new JPanel();
        topPanel.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Model Settings"),
                BorderFactory.createEmptyBorder(0, 5, 5, 5)));
        topPanel.setLayout(new BorderLayout());
        topPanel.add(m_playgroundEditorPanel, BorderLayout.NORTH);

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 0;
        gbC.gridx = 0;
        mainLayout.setConstraints(optionsPanel, gbC);
        mainPanel.add(optionsPanel);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.NORTH;
        gbC.fill = GridBagConstraints.HORIZONTAL;
//        gbC.gridy = 1;
        gbC.gridx = 0;
        mainLayout.setConstraints(buttons, gbC);
        mainPanel.add(buttons);

        gbC = new GridBagConstraints();
        // gbC.anchor = GridBagConstraints.NORTH;
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 2;
        gbC.gridx = 0;
        gbC.weightx = 0;
        mainLayout.setConstraints(historyPanel, gbC);
        mainPanel.add(historyPanel);

        // Setup second column
        JPanel rightPanel = new JPanel();
//        rightPanel.setBorder(BorderFactory.createTitledBorder("Right Panel"));
        GridBagLayout rightLayout = new GridBagLayout();
        rightPanel.setLayout(rightLayout);

        // Add image panel
        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridx = 1;
        gbC.gridy = 0;
        gbC.weightx = 100;
        gbC.weighty = 100;
        gbC.gridheight = 2;
        rightLayout.setConstraints(imagePanel, gbC);
        rightPanel.add(imagePanel);

        // Add output panel
        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 2;
        gbC.gridx = 1;
        gbC.gridheight = 1;
        rightLayout.setConstraints(outputPanel, gbC);
        rightPanel.add(outputPanel);

        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridx = 1;
        gbC.gridy = 0;
        gbC.gridheight = 3;
        gbC.weightx = 100;
        gbC.weighty = 100;
        mainLayout.setConstraints(rightPanel, gbC);
        mainPanel.add(rightPanel);

        setLayout(new BorderLayout());

        add(topPanel, BorderLayout.NORTH);

        add(mainPanel, BorderLayout.CENTER);
    }

    private JPanel setupImagePanel() {
        imagePanel = new JPanel();
        imagePanel.setBorder(BorderFactory.createTitledBorder("Currently Selected Image"));
        imageLabel = new JLabel("", JLabel.CENTER);
        imagePanel.add(imageLabel, BorderLayout.CENTER);

        return imagePanel;
    }

    protected void openNewImage() {
        m_FileChooser.setFileFilter(m_ImageFilter);

        int returnCode = m_FileChooser.showOpenDialog(this);

        if (returnCode == 1) {
            System.err.println("User did not select a new image");
            return;
        }

        File f = m_FileChooser.getSelectedFile();
        m_currentlyDisplayedImage = f.getAbsolutePath();
        refreshImagePanel();
    }

    protected void refreshImagePanel() {
        if (m_currentlyDisplayedImage == null || m_currentlyDisplayedImage == "") {
            return;
        }

        ImageIcon imageIcon = new ImageIcon(m_currentlyDisplayedImage);

        int desiredWidth = imagePanel.getWidth()-50;
        int desiredHeight = imagePanel.getHeight()-50;

        ImageIcon scaledIcon = scaleImage(imageIcon, desiredWidth, desiredHeight);

        imageLabel.setIcon(scaledIcon);

        refreshPredictButtonEnabled();
    }

    protected void refreshPredictButtonEnabled() {
        m_predictButton.setEnabled(true);
    }

    private ImageIcon scaleImage(ImageIcon icon, int desiredWidth, int desiredHeight)
    {
        int newWidth = icon.getIconWidth();
        int newHeight = icon.getIconHeight();

        if(newWidth > desiredWidth)
        {
            newWidth = desiredWidth;
            newHeight = (newWidth * icon.getIconHeight()) / icon.getIconWidth();
        }

        if(newHeight > desiredHeight)
        {
            newHeight = desiredHeight;
            newWidth = (icon.getIconWidth() * newHeight) / icon.getIconHeight();
        }

        return new ImageIcon(icon.getImage().getScaledInstance(newWidth, newHeight, Image.SCALE_DEFAULT));
    }

    private void predict() {
        if (m_RunThread == null) {
            synchronized (this) {
                m_predictButton.setEnabled(false);
            }
            m_RunThread = new Thread() {
                @SneakyThrows
                @Override
                public void run() {
                    m_Logger.statusMessage("Initializing...");
                    Dl4jCNNExplorer explorer = (Dl4jCNNExplorer) m_CNNExplorerEditor.getValue();
                    explorer.setZooModelType(zooModel);
                    try {
                        explorer.init();
                    } catch (Exception ex) {
                        System.err.println("Couldn't initialise model");
                        ex.printStackTrace();
                        return;
                    }

                    m_Logger.statusMessage("Making prediction");
                    explorer.makePrediction(new File(m_currentlyDisplayedImage));

                    m_OutText.setText(explorer.getCurrentPredictions().toSummaryString());

                    synchronized (this) {
                        m_predictButton.setEnabled(true);
                        m_Logger.statusMessage("OK");
                        m_RunThread = null;
                    }
                }
            };
            m_RunThread.setPriority(Thread.MIN_PRIORITY);
            m_RunThread.start();
        }
    }



    /**
     * Updates the enabled status of the input fields and labels.
     */
    protected void updateRadioLinks() {
        m_setModelFileBut.setEnabled(m_modelFileRadio.isSelected());
        m_setZooModelBut.setEnabled(m_ZooModelRadio.isSelected());
    }
}
