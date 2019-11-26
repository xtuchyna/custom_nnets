package pv248;

import java.util.ArrayList;
import java.util.Collection;

public class Neuron {

	double innerPotential = 0;
	double biasWeight;
	double derivativeToErrorWRToutput;
	
	
//	Collection<Neuron> inputs;
	
//	public Neuron(Collection<Neuron> inputs) {
//		this.inputs = inputs;
//	}
	
	/**
	 * Constructor for input Neurons
	 */
	public Neuron() {
		
	}

	public double activationFunction() {
		return 1 / (1 + Math.exp(-innerPotential));
	}


	public double activationFunctionDerivative() {
		return activationFunction()*(1 - activationFunction());
	}
	
	public void refreshInnerPotential(double[] outputs, double[] weights) {
		
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
