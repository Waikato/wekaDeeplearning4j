package weka.dl4j;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import weka.core.CommandlineRunnable;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.converters.AbstractFileLoader;

import java.util.Enumeration;

public class IsGPUAvailable implements CommandlineRunnable, OptionHandler {

    /**
     * Main Entrypoint: Check whether WekaDeeplearning4j can detect a GPU backend.
     * @return true if it can detect a valid GPU backend, false otherwise (either due to no GPU or invalid driver installation)
     */
    public boolean check() {
        boolean result = false;

        Nd4jBackend b = Nd4j.getBackend();
        if (b != null) {
            String backend = b.getClass().getCanonicalName().toLowerCase();
            System.out.printf("Backend is: %s \n", backend);
            result = backend.contains("jcublas");
        }

        return result;
    }

    /**
     * Perform any setup stuff that might need to happen before execution.
     *
     * @throws Exception if a problem occurs during setup
     */
    @Override
    public void preExecution() throws Exception {

    }

    /**
     * Execute the supplied object.
     *
     * @param toRun   the object to execute
     * @param options any options to pass to the object
     * @throws Exception if a problem occurs.
     */
    @Override
    public void run(Object toRun, String[] options) throws Exception {
        boolean result = check();

        System.out.println(result);
    }

    /**
     * Perform any teardown stuff that might need to happen after execution.
     *
     * @throws Exception if a problem occurs during teardown
     */
    @Override
    public void postExecution() throws Exception {

    }

    /**
     * Returns an enumeration of all the available options..
     *
     * @return an enumeration of all available options.
     */
    @Override
    public Enumeration<Option> listOptions() {
        return Option.listOptionsForClass(this.getClass()).elements();
    }

    /**
     * Sets the OptionHandler's options using the given list. All options
     * will be set (or reset) during this call (i.e. incremental setting
     * of options is not possible).
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not sup3ported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        Option.setOptionsForHierarchy(options, this, IsGPUAvailable.class);

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current option settings for the OptionHandler.
     *
     * @return the list of current option settings as an array of strings
     */
    @Override
    public String[] getOptions() {
        return Option.getOptions(this, IsGPUAvailable.class);
    }
}
