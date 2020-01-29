package pv248;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;

public class Main {
	
	public static int MNIST_INPUT_PIXELS = 784;
	public static int MNIST_NUM_OF_LABELS = 10;
	public static int MNIST_TRAIN_DATASET_SIZE = 60000;
	public static int MNIST_TEST_DATASET_SIZE = 10000;
	
	public static int[] mnistLayerScheme = {MNIST_INPUT_PIXELS, 60, 60, 60, MNIST_NUM_OF_LABELS};

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
		
		try {
			System.out.println("Making predictions upon test dataset...");
			predictAndWriteResutls(mnistNeuralNetwork, "actualTestPredictions", inputsTest, labelsTest);
			
			System.out.println("Making predictions upon train dataset...");
			predictAndWriteResutls(mnistNeuralNetwork, "trainPredictions", mnistNeuralNetwork.inputs, mnistNeuralNetwork.outputs);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		Instant finish = Instant.now();
		System.out.println("Overall ellapsed time: " + Duration.between(start, finish).toMinutes());
	}
	
	public static void predictAndWriteResutls(NeuralNetwork network, String filename, int[][] inputs, int[][] expectedOutputs) throws IOException {
		double successfulPredictions = 0;
		
		BufferedWriter writer = new BufferedWriter(new FileWriter(filename));
		
		for(int i = 0; i < inputs.length; i++) {
			int prediction = (int) network.compute(inputs[i]);
			writer.write(String.valueOf(prediction));
			writer.newLine();
			
			if(prediction == expectedOutputs[i][0]) {
				successfulPredictions += 1;
			}
		}
		writer.flush();
		writer.close();
		
		double successRatio = successfulPredictions / inputs.length;
		System.out.println("DONE, succesful predictions: " + successfulPredictions);
		System.out.println("Network ACCURACY (success ratio): " + successRatio*100 + "%");
		
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
