package weka.classifiers.lazy;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Parent abstract k-NN class to be used as part of CO3091's coursework2. 
 * It contains some of the methods necessary to implement k-NN, in order to facilitate your implementation. 
 * You will need to create another class called MyKnn extending KnnParent.
 * Your class MyKnn should implement KnnParent's abstract methods. 
 * 
 * @author Leandro L. Minku (leandro.minku@leicester.ac.uk)
 */
public abstract class KnnParent extends AbstractClassifier implements Classifier, OptionHandler {

	/**
	 * Default serial version
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Parameter k of the nearest neighbour algorithm.
	 */
	protected int m_k;

	/**
	 * Training data used to build the predictive model.
	 * The class Instances is a class to store several instances. It can be used to store the instances
	 * that compose a data set, which is the case here. Each instance held by the Instances class is 
	 * an object of the class Instance. You can find further information about these classes in WEKA's API:
	 * http://weka.sourceforge.net/doc.dev/
	 */
	protected Instances m_TrainingData;
	
	/**
	 * Array for storing the minimum value for each numerical input attribute.
	 * These are the minimum values used in the normalisation 
	 * formula explained in lecture 16 (Introduction to machine learning and k-NN). <p>
	 * 
	 * The size of min[] equals to the number of attributes, i.e., even though 
	 * categorical input attributes and output attributes do not require 
	 * a min value to be determined, they still have
	 * a position allocated to them in the vector min[]. The value that you store in the position
	 * corresponding to categorical input attributes or output attributes does not matter, 
	 * i.e., you can store anything you like in the min[] positions corresponding to them.
	 */
	protected double min[];
	
	/**
	 * Array for storing the maximum value for each numerical input attribute.
	 * These are the maximum values used in the normalisation 
	 * formula explained in lecture 16 (Introduction to machine learning and k-NN). <p>
	 * 
	 * The size of max[] equals to the number of attributes, i.e., even though 
	 * categorical input attributes and output attributes do not require 
	 * a max value to be determined, they still have
	 * a position allocated to them in the vector max[]. The value that you store in the position
	 * corresponding to categorical input attributes or output attributes does not matter, 
	 * i.e., you can store anything you like in the max[] positions corresponding to them.
	 */
	protected double max[];


	/**
	 * K-NN classifier implemented as part of CO3091 coursework 2. <p>
	 *
	 * @param k the number of nearest neighbors to use for prediction
	 */
	public KnnParent(int k) {
		super();
		setK(k);
		min = null;
		max = null;
	}  

	/**
	 * 1-NN classifier implemented as part of CO3091 coursework 2.
	 */
	public KnnParent() {
		super();
		setK(1);
		min = null;
		max = null;
	}

	/**
	 * Returns a string describing classifier.
	 * 
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "K-nearest neighbours classifier "
				+ "implemented for CO3091's coursework 2.\n";
	}

	/**
	 * Set the number of neighbours to use.
	 * If k is less than 1, it will set m_k to 1.
	 *
	 * @param k the number of neighbours.
	 */
	public void setK(int k) {
		if (k < 1) {
			System.out.println("Warning: k must be > 0. Changing it from " + k + " to 1.");
			m_k = 1;
		} else 
			m_k = k;
	}

	/**
	 * Gets the number of nearest neighbours to be used.
	 *
	 * @return the number of neighbours.
	 */
	public int getK() {

		return m_k;
	}

	/**
	 * Returns the tip text for this parameter.
	 * 
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String kTipText() {
		return "Number of nearest neighbours to use.";
	}

	/**
	 * Returns default capabilities of the classifier.
	 * This classifier will only deal with numerical and categorical input attributes and
	 * numerical classes (i.e., regression problems).
	 * The minimum number of examples in the training set is 1. 
	 *
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		//result.enable(Capability.DATE_ATTRIBUTES);
		//result.enable(Capability.MISSING_VALUES);

		// class
		//result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		//result.enable(Capability.DATE_CLASS);
		//result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(1);

		return result;
	}


	/**
	 * Returns an enumeration describing the available options (parameters of the algorithm).
	 *
	 * @return an enumeration of all the available options.
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	@Override
	public Enumeration listOptions() {

		Vector newVector = new Vector(1);

		newVector.addElement(new Option(
				"\tNumber of nearest neighbours to use.\n",
				"-K", 1, "-K 'Number of nearest neighbours to use.'"));

		Enumeration enu = super.listOptions();
		while (enu.hasMoreElements()) {
			newVector.addElement(enu.nextElement());
		}

		return newVector.elements();
	}


	/**
	 * Parses a given list of options (parameters of the algorithm).
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		setK(Integer.parseInt(Utils.getOption('K', options)));
		super.setOptions(options);

	}


	/**
	 * Gets the current parameter settings of the classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String [] getOptions() {
		
		  String [] superOptions = super.getOptions();
		  String [] options = new String [superOptions.length + 2];

		  int current = 0;
		
		  options[current++] = "-K"; 
		  options[current++] = Integer.toString(m_k);
		  
		  System.arraycopy(superOptions, 0, options, current, superOptions.length);
		  current += superOptions.length;

		  return options;
	}

	/**
	 * Return a description of this classifier.
	 *
	 * @return a description of this classifier as a string.
	 */
	public String toString() {

		if (m_TrainingData == null) {
			return "MyKnn: no model built yet.";
		}

		if (m_TrainingData.numInstances() == 0) {
			return "Warning: no training instances in k-NN's model.";
		}

		String result = "MyKnn classifier using k=" + m_k + ".\n" +
				"Trained on " + m_TrainingData.numInstances() + " examples.\n";

		return result;
	}
	
	/**
	 * Build a k-NN model. <p>
	 * 
	 * This consists in copying the trainingData to a field named m_TrainingData,
	 * which represents k-NN's model; determining the min and max values for each 
	 * numerical input attribute; and normalising the m_TrainingData. <p>
	 * 
	 * You will need to implement the methods determineMinMaxAttributeValues
	 * and normaliseNumericInputAttributesTrainingData for this method to work. <p>
	 *  
	 * @param trainingData to be used for building the model.
	 */
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_TrainingData = new Instances(trainingData, 0, trainingData.numInstances());
		determineMinMaxAttributeValues();
		normaliseNumericInputAttributesTrainingData();
	}
	

	/**
	 * Predict the output of a given instance. <p>
	 * 
	 * This consists in normalising the numerical input attributes of that instance,
	 * finding the nearest neighbours of that instance, and 
	 * determining the predicted output (based on the average output of
	 * the nearest neighbours). <p>
	 * 
	 * You will need to implement the methods normaliseNumericInputAttributes,
	 * findNearestNeighbours and determinePredictedOutput for this method to work. <p>
	 * 
	 * PS: WEKA requires this method to be named classifyInstance, even though
	 * this method is used for regression problems in this coursework.
	 * 
	 * @param instance to be predicted.
	 */
	@Override
	public double classifyInstance(Instance instance) {

		normaliseNumericInputAttributes(instance);
		Instance []nearestNeighbours = findNearestNeighbours(instance);
		
		return determinePredictedOutput(nearestNeighbours);
	}
	
	/**
	 * Find the k nearest neighbours of a given instance and store them
	 * in an Instance[] array. <p>
	 * 
	 * If the size of m_TrainingData is less than k,
	 * return all training instances from m_TrainingData.  <p> 
	 * 
	 * If there are any two instances instA and instB with the same distance to "instance"
	 * and the two of them cannot be both returned as nearest neighbours because
	 * the number of neighbours k would be exceeded, then favour the instance
	 * with the smallest index (the former instance) in the data set.  <p>
	 * 
	 * For example, consider that k = 1 and you have a data set with the following instances:  <p>
	 * 
	 * 0, 0, output1
	 * 0.5, 0.5, output2
	 * 0.5, 0.5, output3
	 * 1, 1, output1
	 * ...  <p>
	 * 
	 * You will need to choose between the second and third instances when you return the nearest
	 * neighbour. In this case, you would return the second instance as the nearest neighbour,
	 * because it appears first in the data set.  <p>
	 * 
	 * ******This method must be implemented and overridden by MyKnn ****** <p>
	 * 
	 * TESTING: if you use the whole desharnais data set provided for this 
	 * coursework as the training set, you should find that the 2 nearest neighbours 
	 * of the first instance of the training set are: <p>
	 * 
	 * 1) the instance itself <p>
	 * 2) the instance whose true output is 14434.
	 * 
	 * @param instance whose nearest neighbours are to be found
	 * @return array containing the k nearest neighbours
	 */
	protected abstract Instance[] findNearestNeighbours(Instance instance);
	
	
	/**
	 * Method for normalising the numerical input attributes of all instances
	 * in m_TrainingData. You will need to provide the implementation
	 * for this method based on the normalisation formula explained in
	 * lecture 16 (Introduction to machine learning and k-NN). <p>
	 * 	  
	 * This method assumes that the min and max arrays have already been
	 * set with the appropriate minimum and maximum values for numerical 
	 * input attributes. <p>
	 * 
	 * This method could make use of the method normaliseInputAttributes. <p>
	 * 
	 * ******This method must be implemented and overridden by MyKnn ****** <p>
	 * 
	 * TESTING: if you use the whole desharnais data set provided for this 
	 * coursework as the training set, you should find that the normalised
	 * input attributes for the first and last training examples are, respectively: <p>
	 * 
	 * 0.25, 0.5714285714285714, 0.5, 0.2782212086659065, 0.11842105263157894, 0.22011385199240988, 0.6170212765957447, 0.22770398481973433 <p>
	 * 
	 * and <p>
	 * 
	 * 1.0, 0.5714285714285714, 0.5, 1.0, 0.6157894736842106, 1.0, 0.6170212765957447, 1.0 <p>
	 * 
	 * PS: you may have some very slight difference in the precision of the numbers above. 
	 */
	protected abstract void normaliseNumericInputAttributesTrainingData();
	
	/**
	 * Method for normalising the input attributes of a given instance.
	 * You will need to provide the implementation
	 * for this method based on the normalisation formula explained in
	 * lecture 16 (Introduction to machine learning and k-NN). <p>
	 * 	 
	 * This method assumes that the min and max arrays have already been set. <p>
	 * 
	 * ******This method must be implemented and overridden by MyKnn ****** <p>
	 * 
	 * TESTING: if you use the whole desharnais data set provided for this 
	 * coursework as the training set, you should find that the normalised
	 * input attributes for the first and last training examples are, respectively: <p>
	 * 
	 * 0.25, 0.5714285714285714, 0.5, 0.2782212086659065, 0.11842105263157894, 0.22011385199240988, 0.6170212765957447, 0.22770398481973433 <p>
	 * 
	 * and <p>
	 * 
	 * 1.0, 0.5714285714285714, 0.5, 1.0, 0.6157894736842106, 1.0, 0.6170212765957447, 1.0 <p>
	 * 
	 * PS: you may have some very slight difference in the precision of the numbers above.  
	 * 
	 * @param instance whose attribute is to be normalised
	 */
	protected abstract void normaliseNumericInputAttributes(Instance instance);
	
	/**
	 * Method for setting the min and max arrays with the corresponding
	 * minimum and maximum values of for each numerical input attribute of m_TrainingData. <p>
	 * 
	 * These are the minimum and maximum values used in the normalisation 
	 * formula explained in lecture 16 (Introduction to machine learning and k-NN). <p>
	 * 
	 * ******This method must be implemented and overriden by MyKnn ****** <p>
	 * 
	 * TESTING: if you use the whole desharnais data set provided for this 
	 * coursework as the training set, you should find that the minimum 
	 * and maximum values for the numerical input attributes are the following: <p>
	 * 
	 * min[0]:0.0 <p>
	 * max[0]:4.0 <p>
	 * min[1]:0.0 <p>
	 * max[1]:7.0 <p>
	 * min[2]:82.0 <p>
	 * max[2]:88.0 <p>
	 * min[3]:9.0 <p>
	 * max[3]:886.0 <p>
	 * min[4]:7.0 <p>
	 * max[4]:387.0 <p>
	 * min[5]:73.0 <p>
	 * max[5]:1127.0 <p>
	 * min[6]:5.0 <p>
	 * max[6]:52.0 <p>
	 * min[7]:62.0 <p>
	 * max[7]:1116.0 
	 * 
	 */
	protected abstract void determineMinMaxAttributeValues();
	
	/**
	 * Determine the Euclidean distance between the input attributes of two instances. <p>
	 * For categorical input attributes, use the strategy explained in lecture
	 * 16 (Introduction to machine learning and k-NN) to determine the difference between
	 * two categorical values.
	 *  
	 * ******This method must be implemented and overriden by MyKnn ****** <p>
	 * 
	 * TESTING: if you use the whole desharnais data set provided for this 
	 * coursework as the training set, you should find that the Euclidean distance
	 * between the first and second normalised instances is: <p>
	 * 
	 * 0.6764771002541564. <p>
	 * 
	 * The distance between an instance and itself should be zero. <p>
	 *
	 * The distance between the first and the seventh normalised instances is: <p>
	 * 
	 * 1.160919346197088 <p>
	 * 
	 * PS: you may have some very slight difference in the precision of the numbers above.
	 * 
	 * @param instance1 for which the distance is to be calculated
	 * @param instance2 for which the distance is to be calculated
	 * @return euclidean distance between instance1 and instance2
	 */
	protected abstract double euclideanDistance(Instance instance1, Instance instance2);
	
	/**
	 * Determine the predicted output based on the nearest neighbours. <p>
	 * 
	 * This should be the average of the outputs of the  
	 * nearest neighbours, as explained in lecture 16 
	 * (Introduction to machine learning and k-NN).  <p>
	 * 
	 * If nearestNeighbours contains less elements than k, this method
	 * should still determine the predicted output based on all instances
	 * available in nearestNeighbours.
	 *  
	 * ******This method should be overridden by MyKnn class.****** <p>
	 * 
	 *  TESTING: if you use the whole desharnais data set provided for this 
	 * coursework as the training set, and the nearest neighbours were the first
	 * three instances of the training set, the output would be 3864.0. 
	 * 
	 * 
	 * @param nearestNeighbours array containing the k nearest neighbours
	 * @return predicted class
	 */
	public abstract double determinePredictedOutput(Instance []nearestNeighbours);

}