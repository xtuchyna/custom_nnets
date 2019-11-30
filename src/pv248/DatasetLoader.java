package pv248;

import java.io.FileReader;
import java.io.IOException;

import com.opencsv.CSVReader;

public class DatasetLoader {

	public static int[][] readCsv(String pathToCsv, int datasetSize) {

		int[][] convertedCsv = new int[datasetSize][];
		
		try(CSVReader reader = new CSVReader(new FileReader(pathToCsv))) {
			String[] line;
			int index = 0;
			while ((line = reader.readNext()) != null) {
				
				int[] arrayConvertedLine = new int[line.length];
				for(int i = 0; i < line.length; i++) {
					arrayConvertedLine[i] = Integer.parseInt(line[i]);
				}				
				
				convertedCsv[index] = arrayConvertedLine;
				index++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return convertedCsv;
	}
	
	
}
