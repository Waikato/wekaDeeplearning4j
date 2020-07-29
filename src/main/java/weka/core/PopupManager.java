package weka.core;

import lombok.extern.log4j.Log4j2;

import javax.swing.*;
import java.awt.*;

@Log4j2
public class PopupManager {

    protected JFrame currentFrame;

    /**
     * Checks the stacktrace for a call to GenericObjectEditor.
     * If that exists, WEKA was started by the GUIChooser
     */
    public static void checkIfRunByGUI() {
        // Don't need to check if we've already set this
        if (System.getProperty("weka.started.via.GUIChooser") != null) {
            return;
        }

        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
        System.setProperty("weka.started.via.GUIChooser", "false");
        for (StackTraceElement s : stack) {
            if (s.getClassName().equals("weka.gui.GenericObjectEditor")) {
                System.setProperty("weka.started.via.GUIChooser", "true");
                log.debug("Weka started by GUIChooser");
                break;
            }
        }
    }

    public void createEmptyJFrame(String titleText) {
        checkIfRunByGUI();

        String msg = "Downloading model weights and initializing model, please wait...";
        log.info(msg);
        if (!GraphicsEnvironment.isHeadless() && System.getProperty("weka.started.via.GUIChooser").equals("true")) {
            currentFrame = new javax.swing.JFrame("WekaDeeplearning4j Notification: " + msg);
            currentFrame.setPreferredSize(new java.awt.Dimension(850, 0));
            currentFrame.pack();
            currentFrame.setLocationRelativeTo(null);
            currentFrame.setVisible(true);
            currentFrame.setAutoRequestFocus(true);
            currentFrame.setAlwaysOnTop(true);
        }
        return currentFrame;
    }


}
