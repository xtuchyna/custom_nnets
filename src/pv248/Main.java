package pv248;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import com.opencsv.CSVReader;

public class Main {

	
	public static int MNIST_INPUT_PIXELS = 784;
	public static int MNIST_NUM_OF_LABELS = 10;
	public static int[] mnistLayerScheme = {MNIST_INPUT_PIXELS, 10, 30, 10, 10, MNIST_NUM_OF_LABELS};

	public static int MINI_BATCH_SIZE = 1;
	
	
	public static void main(String args[]) {
		ArrayList<int[]> inputs = readCsv(trainInputs);
		ArrayList<int[]> labels = readCsv(trainLabels);
		
		int[] shuffledIndices = new int[inputs.size()];
		
		for(int i = 0; i < shuffledIndices.length; i++) {
			shuffledIndices[i] = i;
		}
		
		fisherYatesShuffle(shuffledIndices);
		
		
		System.out.println("Generating new network...");
		NeuralNetwork mnistNeuralNetwork = new NeuralNetwork(mnistLayerScheme);
		
		System.out.println("Training network on minibatches...");
		int miniBatchIndex = 0;
		for(int i = 0; i < inputs.size(); i++) {
			
//			int[] oneHotVecLabels = new int[MNIST_NUM_OF_LABELS];
//			oneHotVecLabels[labels.get(shuffledIndices[i])[0]] = 1;
			
			mnistNeuralNetwork.accumulateMiniBatch(inputs.get(shuffledIndices[i]), labels.get(shuffledIndices[i])[0], miniBatchIndex);
			miniBatchIndex++;
			if(miniBatchIndex % MINI_BATCH_SIZE == 0) {
				System.out.print("(" + i + "th input) ");
				mnistNeuralNetwork.train(1);
				miniBatchIndex = 0;
			}
		}
		
		System.out.println("Loading test dataset...");
		ArrayList<int[]> inputsTest = readCsv(testInputs);
		ArrayList<int[]> labelsTest = readCsv(testLabels);
		
//		System.out.println("Testing dataset...");
//		int successfulPredictions = 0;
//		for(int i = 0; i < inputsTest.size(); i++) {
//			int prediction = mnistNeuralNetwork.compute(inputsTest.get(i));
//			if(prediction == labelsTest.get(i)[0]) {
//				successfulPredictions += 1;
//			}
//		}
//		double successRatio = successfulPredictions / inputsTest.size();
//		System.out.println("DONE, succesful predictions:" + successfulPredictions);
//		System.out.println("ACCURACY (success ratio):" + successRatio);
		
		
		
		
		
//		xor.train();
		
//		for(double[][] trainingExample : TAU_XOR ) {
//			System.out.print("For binary input " + trainingExample[0][0] + "," + trainingExample[0][1]);
//
//			double output = xor.compute(trainingExample[0]);
//			System.out.println(" output " + output + " was given.");
//		}
		
		
	}
	
	public static final String currentDir = System.getProperty("user.dir");
	public static final String mnistDataDir = currentDir + "/pv021_project/MNIST_DATA";
	
	public static final String trainInputs = mnistDataDir + "/mnist_train_vectors.csv";
	public static final String trainLabels = mnistDataDir + "/mnist_train_labels.csv";
	
	public static final String testInputs = mnistDataDir + "/mnist_test_vectors.csv";
	public static final String testLabels = mnistDataDir + "/mnist_test_labels.csv";
	
	
	public static ArrayList<int[]> readCsv(String pathToCsv) {
		String csvFile = pathToCsv;

		
		
		ArrayList<int[]> convertedCsv = new ArrayList<int[]>();
		CSVReader reader = null;
		try {
			reader = new CSVReader(new FileReader(csvFile));
			String[] line;
			while ((line = reader.readNext()) != null) {
				ArrayList<Integer> convertedLine = new ArrayList<Integer>();
				int arrSize = 0;
				for (String number : line) {
					convertedLine.add(Integer.valueOf(number));
					arrSize++;
				}
				int[] arrayConvertedLine = new int[arrSize];
				for(int i = 0; i < convertedLine.size(); i++) {
					arrayConvertedLine[i] = convertedLine.get(i);
				}
				convertedCsv.add(arrayConvertedLine);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return convertedCsv;
	}
	
	
	static void fisherYatesShuffle(int[] ar) {
		Random rnd = ThreadLocalRandom.current();
		for (int i = ar.length - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
	}
	
	public static final int[] XOR_LAYER_SCHEME = {2,20,10,1};
	
	public static final double TAU_XOR[][][] = {
			{ {0,0}, {0} },
			{ {0,1}, {1} },
			{ {1,0}, {1} },
			{ {1,1}, {0} }};
	
}
