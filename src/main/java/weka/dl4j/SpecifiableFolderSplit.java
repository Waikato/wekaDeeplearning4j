package weka.dl4j;

import java.io.File;
import java.net.URI;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.canova.api.split.FileSplit;

/**
 * A hacky override of FileSplit that lets you specify
 * the files directly.
 * @author cjb60
 */
public class SpecifiableFolderSplit extends FileSplit {

	private static final long serialVersionUID = 462115492403223134L;
	
    public SpecifiableFolderSplit() {
        super(null, null, true, null, false); // true
    }
    
    public void setFiles(URI[] uris) {
    	locations = uris;
    }
    
    public void setLength(int len) {
    	length = len;
    }
	
}
