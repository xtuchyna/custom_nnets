package pv248;

import java.util.ArrayList;
import java.util.Collection;

public class Neuron {

	double innerPotential = 0;
	double biasWeight = 0;
	double derivativeToErrorWRToutput;
	double weightZeroGradient = 0;
	double prevWeightZeroGradient = 0;
	boolean isOutput = false;
	Neuron[] lastLayer;
	
//	Collection<Neuron> inputs;
	
//	public Neuron(Collection<Neuron> inputs) {
//		this.inputs = inputs;
//	}
	
	/**
	 * Constructor for input Neurons
	 */
	
	public Neuron(boolean isOutput) {
		this.isOutput = isOutput;
	}

	public double softmaxActivation() {
		double expSum = 0;
		for(Neuron neuron : lastLayer) {
			expSum += Math.exp(neuron.innerPotential);
		}
		return Math.exp(innerPotential) / expSum;
	}
	
	public double activationFunction() {
		if(isOutput) {
			return softmaxActivation();
		}
		return 1 / (1 + Math.exp(-(innerPotential)));
	}


	public double activationFunctionDerivative() {
//		if(isOutput) {
//			return ;
//		}
		return activationFunction()*(1 - activationFunction());
	}
	

	public void accumulateWeightWithInputToInnerPotential(double weight, double input) {
		this.innerPotential += weight * input;
	}

	
	public String toString() {
		return String.join("\n", "innPot:" + String.valueOf(this.innerPotential),
								"bias:" + String.valueOf(this.biasWeight),
								"dEk/dy:" + String.valueOf(this.derivativeToErrorWRToutput)) + "\n\n";
	}
	
	/**
	 * We'll see how we use this
	 */
//	public void refreshInnerPotential(Collection<Double> inputs) {
//		innerPotential = inputs.stream()
//				.reduce(Double.valueOf(0), Double::sum);
//		innerPotential += biasWeight;
//	}
	
//	public void calculateInnerPotential() {
//		innerPotential = inputs.stream()
//				.map(neuron -> neuron.activationFunction())
//				.reduce(Double.valueOf(0), Double::sum);
//		innerPotential += biasWeight;
//	}
	
}
