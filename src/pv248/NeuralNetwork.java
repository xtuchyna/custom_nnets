package pv248;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

public class NeuralNetwork {

	double learningRate = 0.1;
	
	/**
	 * Weights are organized in this manner:
	 * weights.get( <index of layer> )[ <index of a neuron which gives output> ][ <index of a neuron which receives the output]
	 * 
	 * Therefore, weight formally written as wij (for any neuron in layer l) is accessed in this way:
	 * weights.get(l)[j][i]
	 * 
	 * To access all weights for a neuron of index n in layer l:
	 * weights.get(l - 1)[n]
	 */
	ArrayList<double[][]> weights;
	ArrayList<double[][]> gradients;
	
	ArrayList<Neuron[]> layers;
	
	public NeuralNetwork(int[] layerScheme) {
		generateNetwork(layerScheme);
		generateWeights(layerScheme);
	}
	
	public void generateWeights(int[] layerScheme) {
		
		// generating weights for each 'inter-connection' of layers
		for(int i = 0; i < layerScheme.length - 1; i++) {
			double[][] curWeight = new double[layerScheme[i]][layerScheme[i+1]];
			double[][] weightGradient = new double[layerScheme[i]][layerScheme[i+1]];
			
			for(int neuron = 0; neuron < curWeight.length; neuron++) {
				for(int nextNeuron = 0; nextNeuron < curWeight[neuron].length; nextNeuron++) {
					curWeight[neuron][nextNeuron] = 0.5;
					weightGradient[neuron][nextNeuron] = 0;
				}
			}
			/**
			 * TODO: initliaze with random numbers
			 * the loops above do it just for value 0.5
			 */
			this.weights.add(curWeight);
			this.gradients.add(weightGradient);
		}
	}

	
	public Neuron[] generateLayer(int size) {
		Neuron[] layer = new Neuron[size];
		for(int i = 0; i < layer.length; i++) {
			layer[i] = new Neuron();
		}
		return layer;
	}
	
	public void generateNetwork(int[] layerScheme) {
		for(int numOfNeurons : layerScheme) {
			this.layers.add(generateLayer(numOfNeurons));
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
		for(int i = 0; i < inputTuple.length; i++) {
			this.layers.get(0)[i].refreshInnerPotential(inputTuple[i]);
		}
		
		for(int layerIndex = 1; layerIndex < this.layers.size(); layerIndex++) {
			Neuron[] curLayer = this.layers.get(layerIndex);
			for(int neuronIndex = 0; neuronIndex < curLayer.length; neuronIndex++) {
				
				curLayer[neuronIndex].refreshInnerPotential(sumInputs(neuronIndex, layerIndex-1));
			}
		}

		//return output from last layer
		//(XOR problem so the output neuron is only one neuron
		//thus one double value and not an array of doubles
		//TODO: change when scaling the problem
		return this.layers.get(this.layers.size()-1)[0].activationFunction();
	}
	
	public double sumInputs(int neuronIndex, int prevLayerIndex) {
		double sum = 0;
		double prevLayer[][] = this.weights.get(prevLayerIndex);

		for(double prevNeuron[] : prevLayer) {
			sum += prevNeuron[neuronIndex];
		}
		return sum;
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
	public double crossEntropy() {
		double acc = 0;
		for(double[][] trainExample : TAU_XOR) {
			double expectedOutput = trainExample[1][0];		ArrayList<double[][]> weightGradients = new ArrayList<double[][]>();

			double computedOutput = compute(trainExample[0]);
			
			acc += expectedOutput * Math.log(computedOutput) + (1 - expectedOutput) * Math.log(1 - computedOutput); 
		}
		
		return -acc;
	}

	public double crossEntropyDerivativeWRT(int output) {
		
		return 0;
	}

	
	public double meanSquareError() {
		double acc = 0;
		for(double[][] trainExample : TAU_XOR) {
			double expectedOutput = trainExample[1][0];
			double computedOutput = compute(trainExample[0]);
			
			acc += Math.pow((computedOutput - expectedOutput), 2); 
		}
		
		return 1/2 * acc; 
	}
	
	
	public double meanSquareErrorDerivativeWRT(int neuronIndex, int layerIndex, double expectedOutput) {
		//if the neuron is in the output layer
		if (layerIndex == this.layers.size()-1) {
			double derivate = this.layers.get(layerIndex)[neuronIndex].activationFunction() - expectedOutput;;
			
			this.layers.get(layerIndex)[neuronIndex].derivativeToErrorWRToutput = derivate;
			
			return derivate;
		}
		
		int layerAbove = layerIndex + 1;
		int sum = 0;
		for(int neuronAbove = 0; neuronAbove < this.weights.get(layerAbove-1).length; neuronAbove++) {
			sum += this.layers.get(layerAbove)[neuronAbove].derivativeToErrorWRToutput
			* this.layers.get(layerAbove)[neuronAbove].activationFunctionDerivative()
			* this.weights.get(layerIndex - 1)[neuronIndex][neuronAbove];
		}
		this.layers.get(layerIndex)[neuronIndex].derivativeToErrorWRToutput = sum;
		
		return sum;
	}
	

	public static final int WOOSH = 5;
	public void train() {
		for(int epoch = 0; epoch < WOOSH; epoch++) {
			gradientDescent();
		}
	}
	
	
	public void gradientDescent() {

		//for every example in training set
		for(double[][] trainingExample : TAU_XOR) {
			
			//forwardpass
			double computedOutput = forwardPass(trainingExample[0]);
			
			//backwardpass
			backwardPass(computedOutput);
			
			//gradient computation and accumulation
			for(int i = 0; i < this.gradients.size()-1; i++) {
				double[][] curLayerGradients = this.gradients.get(i);
				
				for(int j = 0; j < curLayerGradients.length; j++) {
					Neuron weightOwner = this.layers.get(i+1)[j];
					
					//here the derivative of Ek wrt wji is computed and summed to the Eji
					curLayerGradients[i][j] += weightOwner.derivativeToErrorWRToutput
											* weightOwner.activationFunctionDerivative()
											* weightOwner.activationFunction();
				}
			}
		}
		
		for(int interLayer = 0; interLayer < this.weights.size()-1; interLayer++) {
			double[][] curInterLayerWeights = this.weights.get(interLayer); 

			for(int weightOwner = 0; weightOwner < this.layers.get(interLayer+1).length; weightOwner++) {

				for(int weightGiver = 0; weightGiver < curInterLayerWeights.length; weightGiver++) {
					
					//changing each weight
					this.weights.get(interLayer)[weightGiver][weightOwner] += -learningRate*this.gradients.get(interLayer)[weightGiver][weightOwner];
				}
			}
		}
		
	}
	
	
	
	
	public double forwardPass(double[] trainingInput) {
		return compute(trainingInput);
	}
	
	
	/**
	 * For every neuron in network compute derivative of Err(expectedOutput) wrt output yj using backpropagation
	 * 
	 * @param expectedOutput
	 */
	public void backwardPass(double expectedOutput) {
		for(int layerIndex = 0;  layerIndex < this.layers.size(); layerIndex++) {
			Neuron[] curLayer = this.layers.get(layerIndex);
			
			for(int neuronIndex = 0; neuronIndex < curLayer.length; neuronIndex++) {
				curLayer[neuronIndex].derivativeToErrorWRToutput = meanSquareErrorDerivativeWRT(neuronIndex, layerIndex, expectedOutput);
			}
		}
	}
	

}
