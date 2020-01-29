package pv248;

import java.util.ArrayList;
import java.util.Collection;

public abstract class Neuron {

	double innerPotential = 0;
	double biasWeight = 0;
	double derivativeToErrorWRToutput;
	double weightZeroGradient = 0;
	double prevWeightZeroGradient = 0;
	Neuron[] lastLayer;

	public abstract double activationFunction();
	
	public double activationFunctionDerivative() {
		return activationFunction()*(1.0 - activationFunction());
	}

	public void accumulateWeightWithInputToInnerPotential(double weight, double input) {
		this.innerPotential += weight * input;
	}

}
