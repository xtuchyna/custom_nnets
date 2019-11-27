package pv248;

import java.util.ArrayList;
import java.util.Collection;

public class Neuron {

	double innerPotential = 0;
	double biasWeight = 0;
	double derivativeToErrorWRToutput;
	double weightZeroGradient = 0;
	double prevWeightZeroGradient = 0;
	boolean isInputNeuron = false;
	
//	Collection<Neuron> inputs;
	
//	public Neuron(Collection<Neuron> inputs) {
//		this.inputs = inputs;
//	}
	
	/**
	 * Constructor for input Neurons
	 */
	public Neuron() {
		
	}
	
	public Neuron(boolean isInput) {
		this.isInputNeuron = isInput;
	}

	public double activationFunction() {
//		if(isInputNeuron) {
//			return innerPotential;
//		}
		return 1 / (1 + Math.exp(-(innerPotential+biasWeight)));
	}


	public double activationFunctionDerivative() {
//		if(isInputNeuron) {
//			return 1;
//		}
		return activationFunction()*(1 - activationFunction());
	}
	

	public void accumulateWeightWithInputToInnerPotential(double weight, double input) {
		this.innerPotential += weight * input;
	}

	/**
	 * Used only for INPUT NEURONS
	 * @param input
	 */
	public void refreshInnerPotential(double input) {
		this.innerPotential += input;
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
