package weka.classifiers.lazy;

import weka.core.Instance;

/**
 * k-NN class to be implemented by you as part of CO3091's coursework2.
 */

public class MyKnn extends KnnParent {

	public MyKnn(int k) {
		super(k);

	}

	public MyKnn() {
		super();
	}

	@Override
	protected Instance[] findNearestNeighbours(Instance instance) {
		Instance[] nearest = null;
		int smallest = getK();
		double distances[] = new double[Math.min(smallest,
				m_TrainingData.size())];
		nearest = new Instance[Math.min(smallest, m_TrainingData.size())];

		// finds the nearest distances up to k and the number of instances
		for (int i = 0; i < smallest && i < m_TrainingData.numInstances(); ++i) {
			distances[i] = euclideanDistance(instance, m_TrainingData.get(i));
			nearest[i] = m_TrainingData.get(i);
		}
		// finds the nearest distances from k
		for (int i = smallest; i < m_TrainingData.numInstances(); ++i) {
			double distance = euclideanDistance(instance, m_TrainingData.get(i));
			int furthestNearestNeighbourIndex = 0;
			double maxDistance = distances[0];

			// max index, if it appears twice, will take the smallest index
			for (int j = 1; j < distances.length; ++j) {
				if (distances[j] > maxDistance) {
					furthestNearestNeighbourIndex = j;
					maxDistance = distances[j];
				}
			}

			int furthestNearestNeighbour = furthestNearestNeighbourIndex;

			if (distance < distances[furthestNearestNeighbour]) {
				distances[furthestNearestNeighbour] = distance;
				nearest[furthestNearestNeighbour] = m_TrainingData.get(i);
			}
		}
		return nearest;
	}

	@Override
	protected void normaliseNumericInputAttributesTrainingData() {

		for (int i = 0; i < m_TrainingData.numInstances(); i++) {
			Instance instance = m_TrainingData.get(i);
			normaliseNumericInputAttributes(instance);
		}

	}

	@Override
	protected void normaliseNumericInputAttributes(Instance instance) {

		int index = 0;

		for (int i = 0; i < instance.numValues(); i++) {
			if (m_TrainingData.attribute(i).isNumeric()
					&& i != m_TrainingData.classIndex()) {
				//Calculation for normalisation
				double normalised = (instance.value(i) - min[index])
						/ (max[index] - min[index]);
				instance.setValue(i, normalised);
				index++;
			}

		}

	}

	@Override
	protected void determineMinMaxAttributeValues() {
		min = new double[m_TrainingData.instance(0).numAttributes()];
		max = new double[m_TrainingData.instance(0).numAttributes()];

		int index = 0;
		for (int i = 0; i < m_TrainingData.instance(0).numAttributes(); i++) {
			if (i != m_TrainingData.classIndex()) {
				if (m_TrainingData.attribute(i).isNumeric()) {
					double[] attributeArr = new double[m_TrainingData
							.numInstances()];
					for (int n = 0; n < m_TrainingData.numInstances(); n++) {
						attributeArr[n] = m_TrainingData.instance(n).value(i);
					}

					// assign temp values to test for min and max attribute
					// values
					double maxTemp = attributeArr[0];
					double minTemp = attributeArr[0];

					// sorts the array for min and max values to order
					for (int j = 1; j < attributeArr.length; j++) {
						if (attributeArr[j] > maxTemp) {
							maxTemp = attributeArr[j];
						} else if (attributeArr[j] < minTemp) {
							minTemp = attributeArr[j];
						}
					}
					min[index] = minTemp;
					max[index] = maxTemp;

					// loop counter incremented
					index++;
				} else {
					min[index] = 0;
					max[index] = 0;
				}
			}
		}
	}

	@Override
	protected double euclideanDistance(Instance instance1, Instance instance2) {
		double sum = 0;
		for (int i = 0; i < instance1.numValues(); i++) {
			if (instance1.attribute(i).isNumeric()
					&& i != m_TrainingData.classIndex()) {
				sum += Math.pow(instance1.value(i) - instance2.value(i), 2);
			} else if (!instance1.attribute(i).isNumeric()
					&& i != m_TrainingData.classIndex()) {
				if (instance1.value(i) != instance2.value(i)) {
					sum += 1;
				}
			}
		}
		// take the square root of the sum for the euclidean distance
		sum = Math.sqrt(sum);

		return sum;
	}

	@Override
	public double determinePredictedOutput(Instance[] nearestNeighbours) {
		double runningTotal = 0;

		for (int i = 0; i < nearestNeighbours.length; i++) {
			// adds the values of the nearest neighbour array
			runningTotal += nearestNeighbours[i].classValue();
		}
		// takes the average of the running total
		double avg = runningTotal / nearestNeighbours.length;

		return avg;
	}
}