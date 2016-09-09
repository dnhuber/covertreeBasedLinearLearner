package covertree;

import java.io.IOException;
import java.text.ParseException;


public class main{

	public static void main(String[] args) throws IOException, ParseException {

		//first column in trainset has to be the variable that will be predicted later (first column is y, others are x1,x2,...,xn)
		long time= System.currentTimeMillis();
		
		String trainSource=".../ENB2012_y2_train.csv";		//path to trainSet
		String testSource=".../ENB2012_y2_test.csv";		//path to testSet (without y)
		String testOutput=".../ENB2012_y2_result.csv";		//path to designated output file with all forcasts for testSet values
		String delimiter=";";	//delimiter for trainSet and testSet csv
		
		//function to build the Covertree from trainSource, do forecasts for testSource and save results to testOutput
		firstTest.forecast(trainSource, testSource, testOutput, delimiter);
		
		
		System.out.println("Totaltime : "+(System.currentTimeMillis()-time) +"ms");
	}
}