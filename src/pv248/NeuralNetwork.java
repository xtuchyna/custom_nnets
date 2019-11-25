package pv248;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public class NeuralNetwork {

	double[][] hiddenToInputWeights;
	double[][] outputToHiddenWeights;
	
	Neuron[] inputLayer;
	Neuron[] hiddenLayer;
	Neuron[] outputLayer;
	
	public NeuralNetwork(int input, int hidden, int output) {
		this.inputLayer = initLayer(input);
		this.hiddenLayer = initLayer(hidden);
		this.outputLayer = initLayer(output);
		
		hiddenToInputWeights = new double[hidden][input];
		outputToHiddenWeights = new double[output][hidden];
	}
	
	public Neuron[] initLayer(int size) {
		Neuron[] layer = new Neuron[size];
		for(int i = 0; i < layer.length; i++) {
			layer[i] = new Neuron();
		}
		return layer;
	}

	public void computeLayer(Neuron[] prevLayer, Neuron[] curLayer, double[][] weights) {
		for(Neuron curNeuron: curLayer) {
			
			Collection<Pair> inputs = new ArrayList<Double>();
			
			for(Neuron inputNeuron : prevLayer) {
				inputs.add(inputNeuron.activationFunction());
			}
			curNeuron.refreshInnerPotential(inputs);
		}
	}

	/**
	 * for now i assume that my net is dense in means of every two layes
	 * connected within each other with maximal density
	 * @param inputTuple
	 * @return
	 */
	public double compute(double[] inputTuple) {
		//input neurons are computed differently
		for(int i = 0; i <= inputTuple.length; i++) {
			inputLayer[i].refreshInnerPotential(Arrays.asList(inputTuple[i]));
		}
		
		computeLayer(inputLayer, hiddenLayer);
		computeLayer(hiddenLayer, outputLayer);
		
		return outputLayer[0].activationFunction();
	}
	
	public static final double TAU_XOR[][][] = {
			{ {0,0}, {0} },
			{ {0,1}, {1} },
			{ {1,0}, {1} },
			{ {1,1}, {0} }};

	/**
	 * computes log likelihood and then returns cross entropy
	 * cross entropy = -ll
	 */
	public double errorFunction() {
		double acc = 0;
		for(double[][] trainExample : TAU_XOR) {
			double expectedOutput = trainExample[1][0];
			double computedOutput = compute(trainExample[0]);
			
			acc += expectedOutput * Math.log(computedOutput)
					+ (1 - expectedOutput) * Math.log(1 - computedOutput); 
		}
		
		return -acc;
	}
	
	public void train() {
		
	}

}
