package pv248;

public class OutputNeuron extends Neuron {

	@Override
	public double activationFunction() {
//		
//		return 1 / (1 + Math.exp(-(innerPotential)));
//		
		/**
		 * Softmax, just uncomment
		 */
		double expSum = 0;
		for(Neuron neuron : lastLayer) {
			expSum += Math.exp(neuron.innerPotential);
		}
		return Math.exp(innerPotential) / expSum;
	}


}
