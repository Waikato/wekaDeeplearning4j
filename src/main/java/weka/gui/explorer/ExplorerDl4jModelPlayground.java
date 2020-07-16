package weka.gui.explorer;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.evaluation.output.prediction.Null;
import weka.core.*;

import weka.dl4j.playground.Dl4jImageModelPlayground;
import weka.gui.*;
import weka.gui.explorer.Explorer.ExplorerPanel;
import weka.gui.explorer.Explorer.LogHandler;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileFilter;
import java.awt.*;
import java.awt.event.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.io.File;

public class ExplorerDl4jModelPlayground extends JPanel implements ExplorerPanel, LogHandler {

    /** the parent frame */
    protected Explorer m_Explorer = null;

    /** sends notifications when the set of working instances gets changed*/
    protected PropertyChangeSupport m_Support = new PropertyChangeSupport(this);

    protected Instances m_Instances = null;

    protected Logger m_Logger = null;

    protected Dl4jImageModelPlayground m_dl4jImageModelPlayground = new Dl4jImageModelPlayground();

    /**
     * UI Components
     */

    /** The filename extension that should be used for model files. */
    public static String MODEL_FILE_EXTENSION = ".model";

    /** The filename extension that should be used for PMML xml files. */
    public static String PMML_FILE_EXTENSION = ".xml";

    /** Lets the user configure the classifier. */
    protected GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();

    /** The panel showing the current classifier selection. */
    protected PropertyPanel m_CEPanel = new PropertyPanel(m_ClassifierEditor);

    /** The output area for classification results. */
    protected JTextArea m_OutText = new JTextArea(20, 40);

    /** The destination for log/status messages. */
    protected Logger m_Log = new SysErrLog();

    /** A panel controlling results viewing. */
    protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);

    /** Click to set test mode to cross-validation. */
    protected JRadioButton m_CVBut = new JRadioButton("Cross-validation");

    /** Click to set test mode to generate a % split. */
    protected JRadioButton m_PercentBut = new JRadioButton("Percentage split");

    /** Click to set test mode to test on training data. */
    protected JRadioButton m_modelFileRadio = new JRadioButton("Use Model File");

    /** Click to set test mode to a user-specified test set. */
    protected JRadioButton m_ZooModelRadio = new JRadioButton("Zoo Model");

    /** Button for further output/visualize options. */
    JButton m_MoreOptions = new JButton("Open Image...");

    /**
     * Check to save the test data and the predictions in the results list for visualizing later on.
     */
    protected JCheckBox m_StoreTestDataAndPredictionsBut = new JCheckBox(
            "Store test data and predictions for visualization");

    /**
     * Check to collect the predictions for computing statistics such as AUROC.
     */
    protected JCheckBox m_CollectPredictionsForEvaluationBut = new JCheckBox(
            "Collect predictions for evaluation based on AUROC, etc.");

    /**
     * Check to have the point size in error plots proportional to the prediction
     * margin (classification only)
     */
    protected JCheckBox m_errorPlotPointSizeProportionalToMargin = new JCheckBox(
            "Error plot point size proportional to margin");

    /** Check to output the model built from the training data. */
    protected JCheckBox m_OutputModelBut = new JCheckBox("Output model");

    /** Check to output the models built from the training splits. */
    protected JCheckBox m_OutputModelsForTrainingSplitsBut = new JCheckBox(
            "Output models for training splits");

    /** Check to output true/false positives, precision/recall for each class. */
    protected JCheckBox m_OutputPerClassBut = new JCheckBox(
            "Output per-class stats");

    /** Check to output a confusion matrix. */
    protected JCheckBox m_OutputConfusionBut = new JCheckBox(
            "Output confusion matrix");

    /** Check to output entropy statistics. */
    protected JCheckBox m_OutputEntropyBut = new JCheckBox(
            "Output entropy evaluation measures");

    /** Lets the user configure the ClassificationOutput. */
    protected GenericObjectEditor m_ClassificationOutputEditor =
            new GenericObjectEditor(true);

    /** ClassificationOutput configuration. */
    protected PropertyPanel m_ClassificationOutputPanel = new PropertyPanel(
            m_ClassificationOutputEditor);

    /** the range of attributes to output. */
    protected Range m_OutputAdditionalAttributesRange = null;

    /** The button used to open a separate test dataset. */
    protected JButton m_setModelFileBut = new JButton("Set...");

    /** The button used to open a separate test dataset. */
    protected JButton m_setZooModelBut = new JButton("Set...");


    /**
     * Alters the enabled/disabled status of elements associated with each radio
     * button.
     */
    ActionListener m_RadioListener = new ActionListener() {
        @Override
        public void actionPerformed(ActionEvent e) {
            updateRadioLinks();
        }
    };

    /** User specified random seed for cross validation or % split. */
    protected JTextField m_RandomSeedText = new JTextField("1", 3);

    /** the label for the random seed textfield. */
    protected JLabel m_RandomLab = new JLabel("Random seed for XVal / % Split",
            SwingConstants.RIGHT);

    /** Whether randomization is turned off to preserve order. */
    protected JCheckBox m_PreserveOrderBut = new JCheckBox(
            "Preserve order for % Split");


    /** Click to start running the classifier. */
    protected JButton m_StartBut = new JButton("Start");

    /** Click to stop a running classifier. */
    protected JButton m_StopBut = new JButton("Stop");

    /** A thread that classification runs in. */
    protected Thread m_RunThread;

    /** Filter to ensure only model files are selected. */
    protected FileFilter m_ModelFilter = new ExtensionFileFilter(
            MODEL_FILE_EXTENSION, "Model object files");

    protected FileFilter m_PMMLModelFilter = new ExtensionFileFilter(
            PMML_FILE_EXTENSION, "PMML model files");

    /** The file chooser for selecting model files. */
    protected WekaFileChooser m_FileChooser = new WekaFileChooser(new File(
            System.getProperty("user.dir")));

    /**
     * Whether start-up settings have been applied (i.e. initial classifier to
     * use)
     */
    protected boolean m_initialSettingsSet;

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
        JPanel historyHolder = new JPanel(new BorderLayout());
        historyHolder.setBorder(BorderFactory
                .createTitledBorder("Result list (right-click for options)"));
        historyHolder.add(m_History, BorderLayout.CENTER);
        m_ClassifierEditor.setClassType(Classifier.class);
        m_ClassifierEditor.setValue(ExplorerDefaults.getClassifier());
        m_ClassifierEditor.addPropertyChangeListener(new PropertyChangeListener() {
            @Override
            public void propertyChange(PropertyChangeEvent e) {
                m_StartBut.setEnabled(true);
                // Check capabilities
                Capabilities currentFilter = m_ClassifierEditor.getCapabilitiesFilter();
                Classifier classifier = (Classifier) m_ClassifierEditor.getValue();
                Capabilities currentSchemeCapabilities = null;
                if (classifier != null && currentFilter != null
                        && (classifier instanceof CapabilitiesHandler)) {
                    currentSchemeCapabilities =
                            ((CapabilitiesHandler) classifier).getCapabilities();

                    if (!currentSchemeCapabilities.supportsMaybe(currentFilter)
                            && !currentSchemeCapabilities.supports(currentFilter)) {
                        m_StartBut.setEnabled(false);
                    }
                }
                repaint();
            }
        });

        m_modelFileRadio.setToolTipText("Test on the same set that the classifier"
                + " is trained on");
        m_CVBut.setToolTipText("Perform a n-fold cross-validation");
        m_PercentBut.setToolTipText("Train on a percentage of the data and"
                + " test on the remainder");
        m_ZooModelRadio.setToolTipText("Test on a user-specified dataset");
        m_StartBut.setToolTipText("Starts the classification");
        m_StoreTestDataAndPredictionsBut
                .setToolTipText("Store test data and predictions in the result list for later "
                        + "visualization");
        m_CollectPredictionsForEvaluationBut
                .setToolTipText("Collect predictions for calculation of the area under the ROC, etc.");
        m_errorPlotPointSizeProportionalToMargin
                .setToolTipText("In classifier errors plots the point size will be "
                        + "set proportional to the absolute value of the "
                        + "prediction margin (affects classification only)");
        m_OutputModelBut
                .setToolTipText("Output the model obtained from the full training set");
        m_OutputModelsForTrainingSplitsBut
                .setToolTipText("Output the models obtained from the training splits");
        m_OutputPerClassBut.setToolTipText("Output precision/recall & true/false"
                + " positives for each class");
        m_OutputConfusionBut
                .setToolTipText("Output the matrix displaying class confusions");
        m_OutputEntropyBut
                .setToolTipText("Output entropy-based evaluation measures");

        m_RandomLab.setToolTipText("The seed value for randomization");
        m_RandomSeedText.setToolTipText(m_RandomLab.getToolTipText());
        m_PreserveOrderBut
                .setToolTipText("Preserves the order in a percentage split");

        m_FileChooser.addChoosableFileFilter(m_PMMLModelFilter);
        m_FileChooser.setFileFilter(m_ModelFilter);

        m_FileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);

        m_ClassificationOutputEditor.setClassType(AbstractOutput.class);
        m_ClassificationOutputEditor.setValue(new Null());

        m_StoreTestDataAndPredictionsBut.setSelected(ExplorerDefaults
                .getClassifierStoreTestDataAndPredictionsForVis());
        m_CollectPredictionsForEvaluationBut.setSelected(ExplorerDefaults
                .getClassifierCollectPredictionsForEvaluation());
        m_OutputModelBut.setSelected(ExplorerDefaults.getClassifierOutputModel());
        m_OutputModelsForTrainingSplitsBut.setSelected(ExplorerDefaults
                .getClassifierOutputModelsForTrainingSplits());
        m_OutputPerClassBut.setSelected(ExplorerDefaults
                .getClassifierOutputPerClassStats());
        m_OutputConfusionBut.setSelected(ExplorerDefaults
                .getClassifierOutputConfusionMatrix());
        m_OutputEntropyBut.setSelected(ExplorerDefaults
                .getClassifierOutputEntropyEvalMeasures());
        m_RandomSeedText.setText("" + ExplorerDefaults.getClassifierRandomSeed());
        m_PreserveOrderBut.setSelected(ExplorerDefaults
                .getClassifierPreserveOrder());


        m_CVBut.setSelected(true);
        // see "testMode" variable in startClassifier
        m_CVBut.setSelected(ExplorerDefaults.getClassifierTestMode() == 1);
        m_PercentBut.setSelected(ExplorerDefaults.getClassifierTestMode() == 2);
        m_modelFileRadio.setSelected(ExplorerDefaults.getClassifierTestMode() == 3);
        m_ZooModelRadio.setSelected(ExplorerDefaults.getClassifierTestMode() == 4);

        updateRadioLinks();
        ButtonGroup bg = new ButtonGroup();
        bg.add(m_modelFileRadio);
        bg.add(m_CVBut);
        bg.add(m_PercentBut);
        bg.add(m_ZooModelRadio);
        m_modelFileRadio.addActionListener(m_RadioListener);
        m_CVBut.addActionListener(m_RadioListener);
        m_PercentBut.addActionListener(m_RadioListener);
        m_ZooModelRadio.addActionListener(m_RadioListener);
        m_setZooModelBut.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                System.out.println("Opening Zoo model prompt");
//                setTestSet();
            }
        });

        m_setModelFileBut.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                System.out.println("Opening model file prompt...");
            }
        });

        m_StartBut.setEnabled(false);
        m_StopBut.setEnabled(false);
        m_StartBut.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                boolean proceed = true;
                if (Explorer.m_Memory.memoryIsLow()) {
                    proceed = Explorer.m_Memory.showMemoryIsLow();
                }
                if (proceed) {
//                    startClassifier();
                }
            }
        });
        m_StopBut.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
//                stopClassifier();
            }
        });

        m_History.setHandleRightClicks(false);
//        // see if we can popup a menu for the selected result
//        m_History.getList().addMouseListener(new MouseAdapter() {
//            @Override
//            public void mouseClicked(MouseEvent e) {
//                if (((e.getModifiers() & InputEvent.BUTTON1_MASK) != InputEvent.BUTTON1_MASK)
//                        || e.isAltDown()) {
//                    int index = m_History.getList().locationToIndex(e.getPoint());
//                    if (index != -1) {
//                        java.util.List<String> selectedEls =
//                                (java.util.List<String>) m_History.getList().getSelectedValuesList();
//                        // String name = m_History.getNameAtIndex(index);
//                        visualize(selectedEls, e.getX(), e.getY());
//                    } else {
//                        visualize(null, e.getX(), e.getY());
//                    }
//                }
//            }
//        });



        // Layout the GUI
//        JPanel p1 = new JPanel();
//        p1.setBorder(BorderFactory.createCompoundBorder(
//                BorderFactory.createTitledBorder("Classifier"),
//                BorderFactory.createEmptyBorder(0, 5, 5, 5)));
//        p1.setLayout(new BorderLayout());
//        p1.add(m_CEPanel, BorderLayout.NORTH);

        JPanel p2 = new JPanel();
        GridBagLayout gbL = new GridBagLayout();
        p2.setLayout(gbL);
        p2.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("Playground options"),
                BorderFactory.createEmptyBorder(0, 5, 5, 5)));

        GridBagConstraints gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.WEST;
        gbC.gridy = 0;
        gbC.gridx = 0;
        gbL.setConstraints(m_modelFileRadio, gbC);
        p2.add(m_modelFileRadio);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.EAST;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 0;
        gbC.gridx = 1;
        gbC.gridwidth = 2;
        gbC.insets = new Insets(2, 10, 2, 0);
        gbL.setConstraints(m_setModelFileBut, gbC);
        p2.add(m_setModelFileBut);

//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.WEST;
//        gbC.gridy = 1;
//        gbC.gridx = 0;
//        gbL.setConstraints(m_TestSplitBut, gbC);
//        p2.add(m_TestSplitBut);
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.EAST;
//        gbC.fill = GridBagConstraints.HORIZONTAL;
//        gbC.gridy = 1;
//        gbC.gridx = 1;
//        gbC.gridwidth = 2;
//        gbC.insets = new Insets(2, 10, 2, 0);
//        gbL.setConstraints(m_SetTestBut, gbC);
//        p2.add(m_SetTestBut);



        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.WEST;
        gbC.gridy = 1;
        gbC.gridx = 0;
        gbL.setConstraints(m_ZooModelRadio, gbC);
        p2.add(m_ZooModelRadio);

        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.EAST;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 1;
        gbC.gridx = 1;
        gbC.gridwidth = 2;
        gbC.insets = new Insets(2, 10, 2, 0);
        gbL.setConstraints(m_setZooModelBut, gbC);
        p2.add(m_setZooModelBut);


//
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.WEST;
//        gbC.gridy = 2;
//        gbC.gridx = 0;
//        gbL.setConstraints(m_CVBut, gbC);
//        p2.add(m_CVBut);
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.EAST;
//        gbC.fill = GridBagConstraints.HORIZONTAL;
//        gbC.gridy = 2;
//        gbC.gridx = 1;
//        gbC.insets = new Insets(2, 10, 2, 10);
//        gbL.setConstraints(m_CVLab, gbC);
//        p2.add(m_CVLab);
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.EAST;
//        gbC.fill = GridBagConstraints.HORIZONTAL;
//        gbC.gridy = 2;
//        gbC.gridx = 2;
//        gbC.weightx = 100;
//        gbC.ipadx = 20;
//        gbL.setConstraints(m_CVText, gbC);
//        p2.add(m_CVText);
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.WEST;
//        gbC.gridy = 3;
//        gbC.gridx = 0;
//        gbL.setConstraints(m_PercentBut, gbC);
//        p2.add(m_PercentBut);
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.EAST;
//        gbC.fill = GridBagConstraints.HORIZONTAL;
//        gbC.gridy = 3;
//        gbC.gridx = 1;
//        gbC.insets = new Insets(2, 10, 2, 10);
//        gbL.setConstraints(m_PercentLab, gbC);
//        p2.add(m_PercentLab);
//
//        gbC = new GridBagConstraints();
//        gbC.anchor = GridBagConstraints.EAST;
//        gbC.fill = GridBagConstraints.HORIZONTAL;
//        gbC.gridy = 3;
//        gbC.gridx = 2;
//        gbC.weightx = 100;
//        gbC.ipadx = 20;
//        gbL.setConstraints(m_PercentText, gbC);
//        p2.add(m_PercentText);
//
        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.WEST;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 4;
        gbC.gridx = 0;
        gbC.weightx = 100;
        gbC.gridwidth = 3;

        gbC.insets = new Insets(3, 0, 1, 0);
        gbL.setConstraints(m_MoreOptions, gbC);
        p2.add(m_MoreOptions);

        JPanel buttons = new JPanel();
        buttons.setLayout(new GridLayout(2, 2));
        JPanel ssButs = new JPanel();
        ssButs.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

        ssButs.setLayout(new FlowLayout(FlowLayout.LEFT));

        ssButs.add(m_StartBut);
//        ssButs.add(m_StopBut);


        buttons.add(ssButs);

        JPanel p3 = new JPanel();
        p3.setBorder(BorderFactory.createTitledBorder("Classifier output"));
        p3.setLayout(new BorderLayout());
        final JScrollPane js = new JScrollPane(m_OutText);
        p3.add(js, BorderLayout.CENTER);
        js.getViewport().addChangeListener(new ChangeListener() {
            private int lastHeight;

            @Override
            public void stateChanged(ChangeEvent e) {
                JViewport vp = (JViewport) e.getSource();
                int h = vp.getViewSize().height;
                if (h != lastHeight) { // i.e. an addition not just a user scrolling
                    lastHeight = h;
                    int x = h - vp.getExtentSize().height;
                    vp.setViewPosition(new Point(0, x));
                }
            }
        });

        JPanel mondo = new JPanel();
        gbL = new GridBagLayout();
        mondo.setLayout(gbL);
        gbC = new GridBagConstraints();
        // gbC.anchor = GridBagConstraints.WEST;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 0;
        gbC.gridx = 0;
        gbL.setConstraints(p2, gbC);
        mondo.add(p2);
        gbC = new GridBagConstraints();
        gbC.anchor = GridBagConstraints.NORTH;
        gbC.fill = GridBagConstraints.HORIZONTAL;
        gbC.gridy = 1;
        gbC.gridx = 0;
        gbL.setConstraints(buttons, gbC);
        mondo.add(buttons);
        gbC = new GridBagConstraints();
        // gbC.anchor = GridBagConstraints.NORTH;
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 2;
        gbC.gridx = 0;
        gbC.weightx = 0;
        gbL.setConstraints(historyHolder, gbC);
        mondo.add(historyHolder);
        gbC = new GridBagConstraints();
        gbC.fill = GridBagConstraints.BOTH;
        gbC.gridy = 0;
        gbC.gridx = 1;
        gbC.gridheight = 3;
        gbC.weightx = 100;
        gbC.weighty = 100;
        gbL.setConstraints(p3, gbC);
        mondo.add(p3);

        setLayout(new BorderLayout());
//        add(p1, BorderLayout.NORTH);
        add(mondo, BorderLayout.CENTER);

        m_ZooModelRadio.setSelected(true);
        updateRadioLinks();
    }

    /**
     * Updates the enabled status of the input fields and labels.
     */
    protected void updateRadioLinks() {
        m_setModelFileBut.setEnabled(m_modelFileRadio.isSelected());
        m_setZooModelBut.setEnabled(m_ZooModelRadio.isSelected());
    }

    /**
     * updates the capabilities filter of the GOE.
     *
     * @param filter the new filter to use
     */
    protected void updateCapabilitiesFilter(Capabilities filter) {
        Instances tempInst;
        Capabilities filterClass;

        if (filter == null) {
            m_ClassifierEditor.setCapabilitiesFilter(new Capabilities(null));
            return;
        }

        if (!ExplorerDefaults.getInitGenericObjectEditorFilter()) {
            tempInst = new Instances(m_Instances, 0);
        } else {
            tempInst = new Instances(m_Instances);
        }
//        tempInst.setClassIndex(m_ClassCombo.getSelectedIndex());

        try {
            filterClass = Capabilities.forInstances(tempInst);
        } catch (Exception e) {
            filterClass = new Capabilities(null);
        }

        // set new filter
        m_ClassifierEditor.setCapabilitiesFilter(filterClass);

        // Check capabilities
        m_StartBut.setEnabled(true);
//        Capabilities currentFilter = m_ClassifierEditor.getCapabilitiesFilter();
//        Classifier classifier = (Classifier) m_ClassifierEditor.getValue();
//        Capabilities currentSchemeCapabilities = null;
//        if (classifier != null && currentFilter != null
//                && (classifier instanceof CapabilitiesHandler)) {
//            currentSchemeCapabilities =
//                    ((CapabilitiesHandler) classifier).getCapabilities();
//
//            if (!currentSchemeCapabilities.supportsMaybe(currentFilter)
//                    && !currentSchemeCapabilities.supports(currentFilter)) {
//                m_StartBut.setEnabled(false);
//            }
//        }
    }
}
