package pv248;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

	public static void main(String args[]) {
		int[] xorLayerScheme = {2,20,10,1};
		//2 20 10 1
		System.out.println("Generating new network...");
		NeuralNetwork xor = new NeuralNetwork(xorLayerScheme);
		
		System.out.println("Training network...");
		xor.train();
		
		System.out.println("Testing network...");
		for(double[][] trainingExample : TAU_XOR ) {
			System.out.print("For binary input " + trainingExample[0][0] + "," + trainingExample[0][1]);

			double output = xor.compute(trainingExample[0]);
			System.out.println(" output " + output + " was given.");
		}
	}
	
	public static final double TAU_XOR[][][] = {
			{ {0,0}, {0} },
			{ {0,1}, {1} },
			{ {1,0}, {1} },
			{ {1,1}, {0} }};
	
}
