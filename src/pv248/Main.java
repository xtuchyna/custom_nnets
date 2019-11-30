package pv248;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import com.opencsv.CSVReader;

public class Main {

	
	public static int MNIST_INPUT_PIXELS = 784;
	public static int MNIST_NUM_OF_LABELS = 10;
	public static int MNIST_TRAIN_DATASET_SIZE = 60000;
	public static int MNIST_TEST_DATASET_SIZE = 10000;
	
	public static int[] mnistLayerScheme = {MNIST_INPUT_PIXELS, 128, MNIST_NUM_OF_LABELS};

	public static void main(String args[]) {
		Instant start = Instant.now();
		
		System.out.println("Generating new network...");
		NeuralNetwork mnistNeuralNetwork = new NeuralNetwork(mnistLayerScheme);
		System.out.println("Loading train datasets...");
		mnistNeuralNetwork.loadDataset(MNIST_TRAIN_INPUTS_PATH, MNIST_TRAIN_OUTPUTS_PATH, MNIST_TRAIN_DATASET_SIZE);
		
		System.out.println("Training network on minibatches...");
		mnistNeuralNetwork.train(1);
		
		System.out.println("Loading test dataset...");
		int[][] inputsTest = DatasetLoader.readCsv(MNIST_TEST_INPUTS_PATH, MNIST_TEST_DATASET_SIZE);
		int[][] labelsTest = DatasetLoader.readCsv(MNIST_TEST_OUTPUTS_PATH, MNIST_TEST_DATASET_SIZE);
		
		System.out.println("Testing dataset...");
		double successfulPredictions = 0;
		for(int i = 0; i < inputsTest.length; i++) {
			int prediction = (int) mnistNeuralNetwork.compute(inputsTest[i]);
			if(prediction == labelsTest[i][0]) {
				successfulPredictions += 1;
			}
		}
		double successRatio = successfulPredictions / inputsTest.length;
		System.out.println("DONE, succesful predictions: " + successfulPredictions);
		System.out.println("Network ACCURACY (success ratio): " + successRatio*100 + "%");
		
		
		
		
		
		Instant finish = Instant.now();
		System.out.println("Overall ellapsed time: " + Duration.between(start, finish).toMinutes());
	}
	
	public static final String CURRENT_DIR = System.getProperty("user.dir");
	public static final String MNIST_DATA_DIR = CURRENT_DIR + "/pv021_project/MNIST_DATA";
	
	public static final String MNIST_TRAIN_INPUTS_PATH = MNIST_DATA_DIR + "/mnist_train_vectors.csv";
	public static final String MNIST_TRAIN_OUTPUTS_PATH = MNIST_DATA_DIR + "/mnist_train_labels.csv";
	public static final String MNIST_TEST_INPUTS_PATH = MNIST_DATA_DIR + "/mnist_test_vectors.csv";
	public static final String MNIST_TEST_OUTPUTS_PATH = MNIST_DATA_DIR + "/mnist_test_labels.csv";
	
	public static final String TRAIN_PREDICTIONS = "trainPredictions";
	public static final String TEST_PREDICTIONS = "actualTestPredictions";
	
	
	public static final int[] XOR_LAYER_SCHEME = {2,20,10,1};
	
	public static final double TAU_XOR[][][] = {
			{ {0,0}, {0} },
			{ {0,1}, {1} },
			{ {1,0}, {1} },
			{ {1,1}, {0} }};
	
}
