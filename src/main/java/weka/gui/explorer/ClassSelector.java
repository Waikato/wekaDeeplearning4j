package weka.gui.explorer;

import weka.dl4j.inference.PredictionClass;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.util.ArrayList;
import java.util.regex.Pattern;

public class ClassSelector extends JPanel {
    JLabel targetClassIDLabel = new JLabel("Target Class ID:");
    JTextField targetClassIDInput = new JTextField();
    JLabel classNameLabel = new JLabel("  Class Name:");
    JTextField classNameInput = new JTextField();
    JButton patternButton = new JButton("Pattern");
    /** The current regular expression. */
    String m_PatternRegEx = "";

    String[] classMap;


    public ClassSelector(String[] classMap) {
        this.classMap = classMap;
        setup();
    }

    private void setup() {
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
        classNameInput.setColumns(40);
        classNameInput.setEditable(false);

        this.setLayout(new FlowLayout(FlowLayout.LEFT));
        this.add(targetClassIDLabel);
        this.add(targetClassIDInput);
        this.add(classNameLabel);
        this.add(classNameInput);
        this.add(patternButton);
    }

//    public void setClassMap(String[] classMap) {
//        processedExplorer.getModelOutputDecoder().getClasses()
//    }

//    public JPanel getPanel() {
//        return row;
//    }

    private void setTargetClass(int id) {
        targetClassIDInput.setText("" + id);
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
        int classID;
        try {
            classID = Integer.parseInt(targetClassIDText);
        } catch (NumberFormatException ex) {
            JOptionPane.showMessageDialog(null,
                    "Error: Please enter a valid integer value", "Error Message",
                    JOptionPane.ERROR_MESSAGE);
            return;
        }
        String newClassName = classMap[classID];
        classNameInput.setText(newClassName);
    }
}
