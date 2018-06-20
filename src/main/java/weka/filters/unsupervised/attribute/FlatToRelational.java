package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.core.Range;
import weka.core.SparseInstance;
import weka.core.WekaException;
import weka.filters.SimpleBatchFilter;

/**
 * A filter which converts a flatly represented timeseries to a relational representation
 *
 * @author Steven Lang
 */
public class FlatToRelational extends SimpleBatchFilter {

  private static final long serialVersionUID = 271653370775136230L;
  /** Number of variables in the timeseries */
  protected int numVariables = 0;
  /** Attribute selection range */
  protected Range range = new Range("");
  /** Attribute indices to collect for the bag */
  protected int[] attsCollectForBagIdxs;
  /** Attribute indices to keep */
  protected int[] attsKeepIdxs;
  /** Relational attribute index */
  protected int relAttIdx;
  /** Keep non-selected attributes */
  protected boolean keepOtherAttributes = true;

  @Override
  public String globalInfo() {
    return "A filter which converts a flat representation to a relational representation";
  }

  @Override
  protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
    if (numVariables == 0) {
      return new Instances(inputFormat, 0);
    } else if (numVariables < 0) {
      throw new WekaException(
          String.format(
              "Invalid input: number of variables was %s. Must be positive.", numVariables));
    }
    range.setUpper(inputFormat.numAttributes() - 1);
    ArrayList<Attribute> attsOut = new ArrayList<>();
    attsCollectForBagIdxs = range.getSelection();
    if (keepOtherAttributes){
      range.setInvert(!range.getInvert());
      attsKeepIdxs = range.getSelection();
    } else{
      // Keep only class
      attsKeepIdxs = new int[]{inputFormat.classIndex()};
    }

    // Keep these attributes
    for (int idx : attsKeepIdxs) {
      attsOut.add((Attribute) inputFormat.attribute(idx).copy());
    }

    // Collect attributes which will be bagged
    ArrayList<Attribute> attsCollectForBag = new ArrayList<>();
    for (int idx : attsCollectForBagIdxs) {
      attsCollectForBag.add((Attribute) inputFormat.attribute(idx).copy());
    }

    // Check if number of input attributes is a multiple of given number of variables
    if (attsCollectForBag.size() % numVariables != 0) {
      throw new WekaException(
          String.format(
              "The total number of attributes (%d) is not a multiple of the given number of "
                  + "variables (%d)",
              attsCollectForBag.size(), numVariables));
    }

    /*
     Check if all attributes are of the same after each #numVariables attributes
    */
    List<Attribute> types = new ArrayList<>();
    // Get the first #numVariables attribute since they are going to be repeated
    for (int i = 0; i < numVariables; i++) {
      types.add(attsCollectForBag.get(i));
    }

    // Check for each attribute
    for (int i = 0; i < attsCollectForBag.size(); i++) {
      Attribute actual = attsCollectForBag.get(i);
      Attribute expected = types.get(i % types.size());

      if (actual.type() != expected.type()) {
        throw new WekaException(
            String.format(
                "Attribute at position %d was of type <%s>, " + "expected type <%s>",
                i, Attribute.typeToString(actual), Attribute.typeToString(expected)));
      }

      if (actual.type() == Attribute.NOMINAL) {
        if (actual.numValues() != expected.numValues()) {
          throw new WekaException(
              String.format("Attributes do not match after each %d attributes", numVariables));
        }

        for (int k = 0; k < actual.numValues(); k++) {
          if (!actual.value(k).equals(expected.value(k))) {
            throw new WekaException(
                String.format("Attributes do not match after each %d attributes", numVariables));
          }
        }
      }
    }

    ArrayList<Attribute> bagAtts = new ArrayList<>();
    for (int i = 0; i < numVariables; i++) {
      Attribute att = (Attribute) types.get(i).copy();
      bagAtts.add(att);
    }
    Instances bagInsts = new Instances("bagInsts", bagAtts, 0);
    Attribute relAtt = new Attribute("bag", bagInsts);
    attsOut.add(0, relAtt);

    // Store index of relational attribute
    relAttIdx = attsOut.indexOf(relAtt);

    final Instances outputFormat = new Instances("filtered", attsOut, 0);
    outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
    return outputFormat;
  }

  @Override
  protected Instances process(Instances instances) throws Exception {
    if (numVariables == 0) {
      return instances;
    }
    Instances outData = new Instances(getOutputFormat());
    for (Instance instOld : instances) {
      Instance inst;
      if (instOld instanceof DenseInstance) {
        inst = new DenseInstance(outData.numAttributes());
      } else if (instOld instanceof SparseInstance) {
        inst = new SparseInstance(outData.numAttributes());
      } else {
        throw new WekaException("Input instance is neither sparse nor dense!");
      }
      inst.setDataset(outData);
      // Collect attributes to keep
      for (int i = 0; i < attsKeepIdxs.length; i++) {
        int idxKeep = attsKeepIdxs[i];
        inst.setValue(i, instOld.value(idxKeep));
      }

      // Create bag data based on relational instances header
      Instances bagData = new Instances(outData.attribute(relAttIdx).relation());
      // Collect attributes to bag
      Instance bagInstance = null;
      for (int i = 0; i < attsCollectForBagIdxs.length; i++) {
        int idxCollect = attsCollectForBagIdxs[i];
        int idxBag = idxCollect % numVariables;

        if (idxBag == 0) {
          // Add old one
          if (i != 0) {
            bagData.add(bagInstance);
          }
          // Create new instance
          bagInstance = new DenseInstance(bagData.numAttributes());
          bagInstance.setDataset(bagData);
        }

        bagInstance.setValue(idxBag, instOld.value(idxCollect));
      }
      // Add relation to the attribute
      final int val = outData.attribute(relAttIdx).addRelation(bagData);

      // Set relation data
      inst.setValue(relAttIdx, val);
      if (instances.classIndex() >= 0){
        inst.setClassValue(instOld.classValue());
      }
      outData.add(inst);
    }
    return outData;
  }

  /** Returns the Capabilities of this filter. */
  @Override
  public Capabilities getCapabilities() {

    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enableAllClasses();
    result.enable(Capability.MISSING_CLASS_VALUES);
    result.enable(Capability.NO_CLASS);

    return result;
  }

  @OptionMetadata(
    displayName = "number of variables",
    description = "The number of variables in the timeseries (default = 0)",
    commandLineParamName = "numVariables",
    commandLineParamSynopsis = "-numVariables <int>",
    displayOrder = 0
  )
  public int getNumVariables() {
    return numVariables;
  }

  public void setNumVariables(int numVariables) {
    this.numVariables = numVariables;
  }

  @OptionMetadata(
    displayName = "attribute range",
    description =
        "The attributes to transform into a multivariate timeseries (default = \"\")",
    commandLineParamName = "range",
    commandLineParamSynopsis = "-range <string>",
    displayOrder = 1
  )
  public String getRange() {
    return range.getRanges();
  }

  public void setRange(String range) {
    this.range = new Range(range);
  }

  @OptionMetadata(
    displayName = "keep other not selected attributes",
    description =
        "Whether to keep (non-class) attributes that are not selected in -range after filtering (default = true)",
    commandLineParamName = "keepOtherAttributes",
    commandLineParamSynopsis = "-keepOtherAttributes <boolean>",
    displayOrder = 2
  )
  public boolean getKeepOtherAttributes() {
    return keepOtherAttributes;
  }

  public void setKeepOtherAttributes(boolean keepOtherAttributes) {
    this.keepOtherAttributes = keepOtherAttributes;
  }

  @Override
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClassHierarchy(this.getClass(), this.getClass().getSuperclass())
        .elements();
  }

  @Override
  public String[] getOptions() {
    return Option.getOptionsForHierarchy(this, this.getClass().getSuperclass());
  }
}
