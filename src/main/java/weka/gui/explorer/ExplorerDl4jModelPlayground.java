package weka.gui.explorer;

import weka.core.Instances;

import weka.gui.Logger;
import weka.gui.explorer.Explorer.ExplorerPanel;
import weka.gui.explorer.Explorer.LogHandler;

import javax.swing.*;
import java.beans.PropertyChangeSupport;

public class ExplorerDl4jModelPlayground extends JPanel implements ExplorerPanel, LogHandler {

    /** the parent frame */
    protected Explorer m_Explorer = null;

    /** sends notifications when the set of working instances gets changed*/
    protected PropertyChangeSupport m_Support = new PropertyChangeSupport(this);

    protected Instances m_Instances = null;

    protected Logger m_Logger = null;

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
}
