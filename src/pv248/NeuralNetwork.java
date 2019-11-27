package pv248;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

public class NeuralNetwork {

	double learningRate = 0.1;
	double momentumFactor = 0.2;
	
	/**
	 * Weights are organized in this manner:
	 * weights.get( <index of layer> )[ <index of a neuron which gives output> ][ <index of a neuron which receives the output]
	 * 
	 * Therefore, weight formally written as wij (for any neuron in layer l) is accessed in this way:
	 * weights.get(l)[j][i]
	 * 
	 * To access all weights for a neuron of index n in layer l:
	 * weights.get(l - 1)[n]
	 * 
	 * TODO: change this method of storing weights
	 * i should access them by weigh.get(l)[i][j] which is more easy to understand
	 * NOTE: well probably does not matter, in backpropagation it would be other way around then
	 */
	ArrayList<double[][]> weights = new ArrayList<double[][]>();
	
	ArrayList<double[][]> prevGradients = new ArrayList<double[][]>();
	ArrayList<double[][]> gradients = new ArrayList<double[][]>();
	
	ArrayList<Neuron[]> layers = new ArrayList<Neuron[]>();
	
	public NeuralNetwork(int[] layerScheme) {
		generateNetwork(layerScheme);
		generateWeights(layerScheme);
	}
	
	public void train(int numOfEpochs) {
		for(int epoch = 0; epoch < numOfEpochs; epoch++) {
			gradientDescent();
			double err = meanSquareError();
			System.out.println("Epoch number: " + epoch + ". Error function: " + err );
		}
	}
	
	public void train() {
		int epoch = 0;
		while(meanSquareError() > 0.01){
			gradientDescent();
			double err = meanSquareError();
			System.out.println("Epoch number: " + epoch + ". Error function: " + err );
			epoch++;
		}
	}
	
	public void generateWeights(int[] layerScheme) {
		Random rand = new Random();
		// generating weights for each 'inter-connection' of layers
		for(int i = 0; i < layerScheme.length - 1; i++) {
			double[][] curWeight = new double[layerScheme[i]][layerScheme[i+1]];
			double[][] weightGradient = new double[layerScheme[i]][layerScheme[i+1]];
			double[][] prevWeightGradient = new double[layerScheme[i]][layerScheme[i+1]];
			
			for(int neuron = 0; neuron < curWeight.length; neuron++) {
				for(int nextNeuron = 0; nextNeuron < curWeight[neuron].length; nextNeuron++) {
					curWeight[neuron][nextNeuron] = rand.nextGaussian();
					weightGradient[neuron][nextNeuron] = 0;
				}
			}
			/**
			 * TODO: initliaze with random numbers
			 * the loops above do it just for value 0.5
			 */
			this.weights.add(curWeight);
			this.gradients.add(weightGradient);
			this.prevGradients.add(prevWeightGradient);
		}
	}

	
	public Neuron[] generateLayer(int size) {
		Neuron[] layer = new Neuron[size];
		for(int i = 0; i < layer.length; i++) {
			if(i == 0) {
				layer[i] = new Neuron(true);
			} else {
				layer[i] = new Neuron();
			}
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
		Neuron[] inputLayer = this.layers.get(0); 
		for(int i = 0; i < inputTuple.length; i++) {
			inputLayer[i].innerPotential = inputTuple[i];
		}
		
		//hidden and output neurons
		for(int layerIndex = 1; layerIndex < this.layers.size(); layerIndex++) {
			Neuron[] curLayer = this.layers.get(layerIndex);
			
			for(int neuronIndex = 0; neuronIndex < curLayer.length; neuronIndex++) {
				
				double[][] giverLayer = this.weights.get(layerIndex-1);
				double sum = 0;
				for(int weightGiver = 0; weightGiver < giverLayer.length; weightGiver++) {
					sum += giverLayer[weightGiver][neuronIndex] * this.layers.get(layerIndex-1)[weightGiver].activationFunction();
				}
				curLayer[neuronIndex].innerPotential = sum;
			}
		}

		//return output from last layer
		//(XOR problem so the output neuron is only one neuron
		//thus one double value and not an array of doubles
		//TODO: change when scaling the problem
		return this.layers.get(this.layers.size()-1)[0].activationFunction();
	}
	
	public static final double TAU_XOR[][][] = {
			{ {0,1}, {1} },
			{ {0,0}, {0} },
			{ {1,0}, {1} },
			{ {1,1}, {0} }};
			
	/**
	 * computes log likelihood and then returns cross entropy
	 * cross entropy = -ll
	 */
//	public double crossEntropy() {
//		double acc = 0;
//		for(double[][] trainExample : TAU_XOR) {
//			double expectedOutput = trainExample[1][0];		ArrayList<double[][]> weightGradients = new ArrayList<double[][]>();
//
//			double computedOutput = compute(trainExample[0]);
//			
//			acc += expectedOutput * Math.log(computedOutput) + (1 - expectedOutput) * Math.log(1 - computedOutput); 
//		}
//		
//		return -acc;
//	}
//
//	public double crossEntropyDerivativeWRT(int output) {
//		
//		return 0;
//	}

	
	public double meanSquareError() {
		double acc = 0;
		for(double[][] trainExample : TAU_XOR) {
			double expectedOutput = trainExample[1][0];
			double computedOutput = compute(trainExample[0]);
			
			acc += Math.pow((computedOutput - expectedOutput), 2); 
		}
		
		return acc/2; 
	}
	
	
	public double meanSquareErrorDerivativeWRT(int neuronIndex, int layerIndex, double expectedOutput) {
		//if the neuron is in the output layer
		if (layerIndex == this.layers.size()-1) {
			return this.layers.get(layerIndex)[neuronIndex].activationFunction() - expectedOutput;
		}
		
		Neuron[] layerAbove = this.layers.get(layerIndex + 1);
		double sum = 0;
		for(int neuronAbove = 0; neuronAbove < layerAbove.length; neuronAbove++) {
			//gradient of neuron from other than output layer is:
			sum += layerAbove[neuronAbove].derivativeToErrorWRToutput
					*layerAbove[neuronAbove].activationFunctionDerivative()
					*this.weights.get(layerIndex)[neuronIndex][neuronAbove];
			
		}
		return sum;
	}
	
	public void clearAndCopyGradientMatrix(){
		for(int i = 0; i < this.gradients.size(); i++) {
			for(int j = 0; j < this.gradients.get(i).length; j++){
				for(int k = 0; k < this.gradients.get(i)[j].length; k++){
					this.prevGradients.get(i)[j][k] = this.gradients.get(i)[j][k]; 
					this.gradients.get(i)[j][k] = 0;
				}
			}
		}
	}

	public void gradientDescent() {
		clearAndCopyGradientMatrix();

		//for every example in training set
		for(double[][] trainingExample : TAU_XOR) {
			
			System.out.println(this.layers);
			
			forwardPass(trainingExample[0]);
			
			backwardPass(trainingExample[1][0]);
			
			//gradient computation and accumulation
			for(int weightLayer = 0; weightLayer < this.weights.size(); weightLayer++){
				for(int i = 0; i < this.weights.get(weightLayer).length; i++){
					for(int j = 0; j < this.weights.get(weightLayer)[i].length; j++){

						//compute gradient as follows
						this.gradients.get(weightLayer)[i][j] +=
										this.layers.get(weightLayer+1)[j].derivativeToErrorWRToutput
										* this.layers.get(weightLayer+1)[j].activationFunctionDerivative()
										* this.layers.get(weightLayer)[i].activationFunction();
						
					}
				}
			}
			

			//bias gradients
			for(Neuron[] layer : this.layers) {
				for(Neuron neuron : layer) {
					neuron.prevWeightZeroGradient = neuron.weightZeroGradient;
					neuron.weightZeroGradient += (neuron.derivativeToErrorWRToutput * neuron.activationFunctionDerivative());
				}
			}
			
		}
		
		//weight update
		for(int weightLayer = 0; weightLayer < this.weights.size(); weightLayer++){
			for(int i = 0; i < this.weights.get(weightLayer).length; i++){
				for(int j = 0; j < this.weights.get(weightLayer)[i].length; j++){
					this.weights.get(weightLayer)[i][j] += (-learningRate * this.gradients.get(weightLayer)[i][j])
														 + (-learningRate * this.momentumFactor * this.prevGradients.get(weightLayer)[i][j]);
				}
			}
		}
		
		//bias weight updates
		for(Neuron[] layer : this.layers) { 
			for(Neuron neuron : layer) {
				neuron.biasWeight += (-learningRate * neuron.weightZeroGradient)
									+(-learningRate * this.momentumFactor * neuron.weightZeroGradient);
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
		for(int layerIndex = this.layers.size()-1;  layerIndex >= 0; layerIndex--) {
			Neuron[] curLayer = this.layers.get(layerIndex);
			
			for(int neuronIndex = 0; neuronIndex < curLayer.length; neuronIndex++) {
				curLayer[neuronIndex].derivativeToErrorWRToutput = meanSquareErrorDerivativeWRT(neuronIndex, layerIndex, expectedOutput);
			}
		}
	}
	

}
