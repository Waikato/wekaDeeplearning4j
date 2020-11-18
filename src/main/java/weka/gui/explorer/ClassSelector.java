package weka.gui.explorer;

import weka.dl4j.inference.PredictionClass;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.util.ArrayList;
import java.util.regex.Pattern;

public class ClassSelector {

    private String[] classMap;

    String m_PatternRegEx = "";

    private JPanel parentPanel;

    JLabel targetClassIDLabel = new JLabel("Target Class ID:");
    JTextField targetClassIDInput = new JTextField();
    JLabel classNameLabel = new JLabel("  Class Name:");
    JTextField classNameInput = new JTextField();
    JButton patternButton = new JButton("Pattern");

    public ClassSelector(JPanel parentPanel, String[] classMap, int defaultClassID, int rowNum) {
        this.parentPanel = parentPanel;
        this.classMap = classMap;
        setup(rowNum);
        setTargetClass(defaultClassID);
    }

    private void setup(int rowNum) {
        patternButton.addActionListener(e -> openPatternDialog());

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
        classNameInput.setEditable(false);

        GridBagConstraints gbc = new SaliencyMapWindow.SaliencyMapGBC();

        gbc.gridy = rowNum;

        gbc.gridx = 0;
        parentPanel.add(targetClassIDLabel, gbc);

        gbc.gridx = 1;
        targetClassIDInput.setMinimumSize(new Dimension(50, 28));
        parentPanel.add(targetClassIDInput, gbc);

        gbc.gridx = 2;
        parentPanel.add(classNameLabel, gbc);

        gbc.gridx = 3;
        gbc.weightx = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        parentPanel.add(classNameInput, gbc);

        gbc.gridx = 4;
        gbc.weightx = 0;
        gbc.fill = GridBagConstraints.NONE;
        parentPanel.add(patternButton, gbc);
    }

    public void removeFromParent() {
        parentPanel.remove(targetClassIDLabel);
        parentPanel.remove(targetClassIDInput);
        parentPanel.remove(classNameLabel);
        parentPanel.remove(classNameInput);
        parentPanel.remove(patternButton);
    }

    public void setTargetClass(int id) {
        targetClassIDInput.setText("" + id);
    }

    public int getTargetClass() {
        return Integer.parseInt(targetClassIDInput.getText());
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
                    setTargetClass(matchingClasses.get(0).getClassID());
                } else {
                    PredictionClass selectedClass = selectOneOfNClasses(matchingClasses);
                    setTargetClass(selectedClass.getClassID());
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
        for (int i = 0; i < classMap.length; i++) {
            String tmpClass = classMap[i];
            if (Pattern.matches(pattern, tmpClass)) {
                result.add(new PredictionClass(i, tmpClass));
            }
        }
        return result;
    }

    private void updateClassNameInput() {
        String targetClassIDText = targetClassIDInput.getText();
        if (targetClassIDText.isEmpty()) {
            return;
        }
        if (this.classMap.length == 0) {
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
        if (classID > classMap.length - 1)
            return;

        String newClassName = classMap[classID];
        classNameInput.setText(newClassName);
    }
}
