
package weka.dl4j.iterators.instance;

import java.io.File;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ResizeImageTransform;
import weka.core.Environment;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.dl4j.ArffMetaDataLabelGenerator;
import weka.gui.ProgrammaticProperty;

/**
 * An iterator that loads images and resizes them.
 *
 * @author Steven Lang
 */
public class ResizeImageInstanceIterator extends ImageInstanceIterator {

  /** SerialVersionUID */
  private static final long serialVersionUID = -3310258401133869149L;

  /** New image height */
  protected int resizeHeight = 28;

  /** New image width */
  protected int resizeWidth = 28;

  /** Backing image iterator */
  protected ImageInstanceIterator iii = new ImageInstanceIterator();

  /**
   * Empty constructor for Weka
   */
  public ResizeImageInstanceIterator(){
  }


  @OptionMetadata(
      displayName = "image iterator",
      description = "The actual iterator used to load the images.",
      commandLineParamName = "imageIterator",
      commandLineParamSynopsis = "-imageIterator <ImageInstanceIterator>",
      displayOrder = 1
  )
  public ImageInstanceIterator getImageInstanceIterator() {
    return iii;
  }

  public void setImageInstanceIterator(ImageInstanceIterator iii) {
    this.iii = iii;
  }

  @Override
  @ProgrammaticProperty
  public int getTrainBatchSize() {
    return iii.getTrainBatchSize();
  }

  @Override
  @ProgrammaticProperty
  public void setTrainBatchSize(int trainBatchSize) {
    iii.setTrainBatchSize(trainBatchSize);
  }

  @OptionMetadata(
      displayName = "resize width",
      description = "The width to resize the image to (default = 28).",
      commandLineParamName = "resizeWidth",
      commandLineParamSynopsis = "-resizeWidth <int>",
      displayOrder = 2
  )
  @Override
  public int getWidth() {
    return resizeWidth;
  }

  @Override
  public void setWidth(int width) {
    resizeWidth = width;
  }

  @OptionMetadata(
      displayName = "resize height",
      description = "The height to resize the image to (default = 28).",
      commandLineParamName = "resizeHeight",
      commandLineParamSynopsis = "-resizeHeight <int>",
      displayOrder = 3
  )
  @Override
  public int getHeight() {
    return resizeHeight;
  }

  @Override
  public void setHeight(int height) {
    resizeHeight = height;
  }

  @Override
  @ProgrammaticProperty
  public int getNumChannels() {
    return iii.getNumChannels();
  }

  @Override
  @ProgrammaticProperty
  public void setNumChannels(int numChannels) {
    iii.setNumChannels(numChannels);
  }

  @Override
  @ProgrammaticProperty
  public File getImagesLocation() {
    return iii.getImagesLocation();
  }

  @Override
  @ProgrammaticProperty
  public void setImagesLocation(File imagesLocation) {
    iii.setImagesLocation(imagesLocation);
  }

  /**
   * Default constructor with the new shape
   *
   * @param iii Previous image iterator
   * @param resizeWidth New image width
   * @param resizeHeight New image height
   */
  public ResizeImageInstanceIterator(ImageInstanceIterator iii, int resizeWidth, int resizeHeight) {
    super();
    this.resizeHeight = resizeHeight;
    this.resizeWidth = resizeWidth;
    this.setTrainBatchSize(iii.getTrainBatchSize());
    this.setImagesLocation(iii.getImagesLocation());
    this.setNumChannels(iii.getNumChannels());
  }

  @Override
  protected ImageRecordReader getImageRecordReader(Instances data) throws Exception {
    Environment env = Environment.getSystemWide();
    String resolved = getImagesLocation().toString();
    try {
      resolved = env.substitute(resolved);
    } catch (Exception ex) {
      // ignore
    }
    ArffMetaDataLabelGenerator labelGenerator =
        new ArffMetaDataLabelGenerator(data, resolved);
    ResizeImageTransform rit = new ResizeImageTransform(getWidth(), getHeight());
    ImageRecordReader reader =
        new ImageRecordReader(getHeight(), getWidth(), getNumChannels(), labelGenerator, rit);
    CollectionInputSplit cis = new CollectionInputSplit(labelGenerator.getPathURIs());
    reader.initialize(cis);
    return reader;
  }
}
