package pv248;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;


public class NeuralNetwork {

	/**
	 * HYPER-PARAMETERS
	 */
	double learningRate = 0.01;
	double momentumFactor = 0.8;
	int miniBatchSize = 64;
	boolean useCrossEntropy = false;
	
	double decay = learningRate / Main.MNIST_TRAIN_DATASET_SIZE;
//	double initiallearningRate = learningRate;
	
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
	
	int[][] inputs;
	int[][] outputs;
	int[] shuffledIndices;
	
	public NeuralNetwork(int[] layerScheme) {
		generateNetwork(layerScheme);
		generateWeights(layerScheme);
	}
	
	public void loadDataset(String inputsFilePath, String outputsFilePath, int datasetSize) {
		this.inputs = new int[datasetSize][];
		this.outputs = new int[datasetSize][];
		
		this.inputs = DatasetLoader.readCsv(inputsFilePath, datasetSize);
		this.outputs = DatasetLoader.readCsv(outputsFilePath, datasetSize);
		
		this.shuffledIndices = new int[datasetSize];
		for(int i = 0; i < shuffledIndices.length; i++) {
			shuffledIndices[i] = i;
		}
		fisherYatesShuffle(shuffledIndices);
	}
	
	private void fisherYatesShuffle(int[] ar) {
		Random rnd = ThreadLocalRandom.current();
		for (int i = ar.length - 1; i > 0; i--) {
			int index = rnd.nextInt(i + 1);
			// Simple swap
			int a = ar[index];
			ar[index] = ar[i];
			ar[i] = a;
		}
	}
	
//	public void accumulateMiniBatch(int[] inputs, int label, int i) {
//		double[] convertedDoubles = new double[inputs.length];
//		for(int j = 0; j < inputs.length; j++) {
//			convertedDoubles[j] = inputs[j];
//		}
//		inputs[i] = convertedDoubles;
//		labels[i] = label;
//	}
	
	public void train(int numOfEpochs) {
		
		int[] trainingIndices = new int[this.miniBatchSize];
		int curIndex = 0;
		for(int i = 0; i < inputs.length; i++) {
			trainingIndices[curIndex] = this.shuffledIndices[i];
			curIndex += 1;
			if (curIndex == this.miniBatchSize) {
				for(int epoch = 0; epoch < numOfEpochs; epoch++) {
					gradientDescent(trainingIndices);
					clearAndCopyGradientMatrix();
					double err = errorFunction(trainingIndices);
					System.out.print("(" + i + "-th input)");
					System.out.println("Epoch number: " + epoch + ". Error function: " + err );
				}
				this.learningRate *= (1 / (1 + this.decay * i));
				System.out.println(" [LR: " + this.learningRate + "]");
				curIndex = 0;
			}
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

	
	public Neuron[] generateLayer(int size, boolean isLast) {
		Neuron[] layer = new Neuron[size];
		for(int i = 0; i < layer.length; i++) {
			layer[i] = new Neuron(isLast);
		}
		
		return layer;
	}
	
	public void generateNetwork(int[] layerScheme) {
		boolean isLast = false;
		for(int i = 0; i < layerScheme.length; i++) {
			if(i == layerScheme.length-1) {
				isLast = true;
			}
			this.layers.add(generateLayer(layerScheme[i], isLast));
		}
		
		Neuron[] lastLayer = this.layers.get(this.layers.size()-1);
		for(Neuron neuron : lastLayer) {
			neuron.lastLayer = lastLayer;
		}
	}
	
	
	/**
	 * for now i assume that my net is dense in means of every two layes
	 * connected within each other with maximal density
	 * @param inputTuple
	 * @return
	 */
	public double compute(int[] inputTuple) {
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
				curLayer[neuronIndex].innerPotential = sum + curLayer[neuronIndex].biasWeight;
			}
		}

		//return output from last layer
		//(XOR problem so the output neuron is only one neuron
		//thus one double value and not an array of doubles
		//TODO: change when scaling the problem
		//return this.layers.get(this.layers.size()-1)[0].activationFunction();
		int lastLayerIndex = this.layers.size() - 1;
		double maximum = 0;
		double maximumIndex = 0;
		for(int i = 0; i < this.layers.get(lastLayerIndex).length; i++) {
			double neuronPrediction = this.layers.get(lastLayerIndex)[i].activationFunction();
			if(neuronPrediction > maximum) {
				maximum = neuronPrediction;
				maximumIndex = i;
			}
		}
		return maximumIndex;
	}
			
	public double errorFunction(int[] miniBatchIndices) {
		if (useCrossEntropy) {
			return categoricalCrossEntropy(miniBatchIndices);
		}
		return meanSquareError(miniBatchIndices);
	}
	
	public double categoricalCrossEntropy(int[] miniBatchIndices) {
		double acc = 0;
		for(int i : miniBatchIndices) {
			
			double computedOutput = compute(this.inputs[i]);
			double expectedOutput = this.outputs[i][0];

			double outputLayerSum = 0;
			Neuron[] lastLayer = this.layers.get(this.layers.size() - 1);

			for(Neuron neuron : lastLayer) {

				outputLayerSum += expectedOutput * Math.log(computedOutput);
			}
			
			acc += outputLayerSum;
		}
		return (-1) * acc /miniBatchIndices.length;
	}
	
	public double meanSquareError(int[] miniBatchIndices) {
		double acc = 0;
		for(int i : miniBatchIndices) {
			double expectedOutput = this.outputs[i][0];
			double computedOutput = compute(this.inputs[i]);
			
			acc += Math.pow((computedOutput - expectedOutput), 2); 
		}
		
		return acc/miniBatchIndices.length; 
	}
	
	
	public double errorFunctionDerivativeWRToutput(int neuronIndex, int layerIndex, double expectedOutput) {
		//if the neuron is in the output layer
		if (layerIndex == this.layers.size()-1) {
			
			double[] oneHotVector = new double[Main.MNIST_NUM_OF_LABELS];
			oneHotVector[(int)expectedOutput] = 1;
			
			double outPut = this.layers.get(layerIndex)[neuronIndex].activationFunction();
			
			if(useCrossEntropy) {
				return (-1) * oneHotVector[neuronIndex] / outPut;
			}
			//else MSE
			return outPut - oneHotVector[neuronIndex];

			//TODO dunno if this is a good idea
			//should init one hot vecs way before probably
		}
		
		//if the neuron is not in the output layer
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
	
	/**
	 * Used both for cleaning the old gradience matrix
	 * and copying it for momentum
	 */
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

	public void gradientDescent(int[] miniBatchIndices) {
		clearAndCopyGradientMatrix();

		//for every example in training set
		for(int miniBatchIndex : miniBatchIndices) {
			
//			System.out.println(this.layers);
			
			forwardPass(this.inputs[miniBatchIndex]);
			
			backwardPass(this.outputs[miniBatchIndex][0]);
			
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
									+(-learningRate * this.momentumFactor * neuron.prevWeightZeroGradient);
			}
		}
		
	}
	
	
	
	
	public double forwardPass(int[] trainingInput) {
		return compute(trainingInput);
	}
	
	
	/**
	 * For every neuron in network compute derivative of Err(expectedOutput) wrt output yj using backpropagation
	 * 
	 * @param expectedOutput
	 */
	public void backwardPass(int expectedOutput) {
		for(int layerIndex = this.layers.size()-1;  layerIndex >= 0; layerIndex--) {
			Neuron[] curLayer = this.layers.get(layerIndex);
			
			for(int neuronIndex = 0; neuronIndex < curLayer.length; neuronIndex++) {
				curLayer[neuronIndex].derivativeToErrorWRToutput = errorFunctionDerivativeWRToutput(neuronIndex, layerIndex, expectedOutput);
			}
		}
	}
	

}
