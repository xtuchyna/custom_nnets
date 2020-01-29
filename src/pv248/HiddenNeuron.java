package pv248;

public class HiddenNeuron extends Neuron {

	@Override
	public double activationFunction() {
//		return 1 / (1 + Math.exp(-(innerPotential)));
		return (innerPotential > 0) ? innerPotential  / NeuralNetwork.MAGIC : 0;
	}

	@Override
	public double activationFunctionDerivative() {
//		return activationFunction()*(1 - activationFunction());
		return (innerPotential > 0) ? 1 : 0;
	}
//
//	@Override
//	public void accumulateWeightWithInputToInnerPotential(double weight, double input) {
//		this.innerPotential += weight * input;
//	}

}
