package pv248;

@Deprecated
public interface GeneralNeuron {

	public double activationFunction();

	public double activationFunctionDerivative();

	public void accumulateWeightWithInputToInnerPotential(double weight, double input);
	
}
