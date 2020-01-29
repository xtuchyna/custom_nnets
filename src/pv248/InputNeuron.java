package pv248;

public class InputNeuron extends Neuron {

	@Override
	public double activationFunction() {
		return this.innerPotential;
	}

	@Override
	public double activationFunctionDerivative() {
		return 0;
	}

	@Override
	public void accumulateWeightWithInputToInnerPotential(double weight, double input) {
		// TODO Auto-generated method stub
	}

}
