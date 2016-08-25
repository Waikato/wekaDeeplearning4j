//package weka.dl4j;
//
//import java.awt.image.BufferedImage;
//import java.io.ByteArrayInputStream;
//import java.io.ByteArrayOutputStream;
//import java.io.DataInputStream;
//import java.io.File;
//import java.io.IOException;
//import java.io.InputStream;
//import java.net.URI;
//import java.util.ArrayList;
//import java.util.Collection;
//import java.util.Iterator;
//
//import javax.imageio.ImageIO;
//
//import org.canova.api.io.data.DoubleWritable;
//import org.canova.api.split.InputSplit;
//import org.canova.api.writable.Writable;
//import org.canova.common.RecordConverter;
//import org.canova.image.loader.ImageLoader;
//import org.canova.image.recordreader.BaseImageRecordReader;
//import org.nd4j.linalg.api.ndarray.INDArray;
//
///**
// * ImageRecordReader assumes that your images are separated into different folders,
// * where each folder is a class. E.g. for MNIST, all the 0 images are in a folder
// * called 0/, all the 1 images in 1/, etc. At test time, you will get instances
// * where the class is not known, e.g:<br />
// * image.png,?<br />
// * This is why we've had to make a new image record reader and make it possible
// * that all the images are in one folder, and we specify the classes for those images
// * explicitly. Besides, it is also nice to have all the images sit in the same folder
// * as well.
// * @author cjb60
// */
//public class EasyImageRecordReader extends BaseImageRecordReader {
//
//	private static final long serialVersionUID = 3806752391833402426L;
//
//	private ArrayList<File> m_filenames = null;
//	private ArrayList<String> m_classes = null;
//
//	private Iterator<String> m_classIterator = null;
//
//	public EasyImageRecordReader(int width, int height, int channels,
//			ArrayList<File> filenames, ArrayList<String> classes) throws IOException {
//		m_filenames = filenames;
//		m_classes = classes;
//
//		initialize(null);
//		imageLoader = new ImageLoader(width, height, channels);
//	}
//
//    @Override
//    public void initialize(InputSplit split) {
//    	// TODO: we don't use the split parameter, which is a bit hacky
//		iter = m_filenames.listIterator();
//		m_classIterator = m_classes.listIterator();
//    }
//
//    @Override
//    public Collection<Writable> next() {
//        if(iter != null) {
//            Collection<Writable> ret = new ArrayList<Writable>();
//            File image = (File) iter.next();
//            //System.out.println(image.getAbsolutePath());
//            String classLabel = m_classIterator.next();
//            currentFile = image;
//            if(image.isDirectory())
//                return next();
//            try {
//                BufferedImage bimg = ImageIO.read(image);
//
//                ByteArrayOutputStream os = new ByteArrayOutputStream();
//                ImageIO.write(bimg, "png", os);
//                InputStream is = new ByteArrayInputStream(os.toByteArray());
//
//                //INDArray row = imageLoader.asRowVector(bimg);
//                INDArray row = imageLoader.asRowVector(is);
//
//                ret = RecordConverter.toRecord(row);
//                if(classLabel != "?")
//                    ret.add(new DoubleWritable(Double.parseDouble(classLabel)));
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//            if(iter.hasNext()) {
//                return ret;
//            }
//            // what is this??
//            /*
//            else {
//                if(iter.hasNext()) {
//                    try {
//                        ret.add(new Text(FileUtils.readFileToString((File) iter.next())));
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }
//                }
//            }
//            */
//            return ret;
//        }
//        else if(record != null) {
//            hitImage = true;
//            return record;
//        }
//        throw new IllegalStateException("No more elements");
//    }
//
//    @Override
//    public void reset() {
//        initialize(null);
//    }
//
//	@Override
//	public Collection<Writable> record(URI uri, DataInputStream dataInputStream)
//			throws IOException {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//}
