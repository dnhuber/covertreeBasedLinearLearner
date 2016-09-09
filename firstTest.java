package covertree;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Timestamp;
import java.text.NumberFormat;
import java.text.ParseException;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;

public class firstTest {

	public static void forecast(String trainSource,String testSource,String testOutput, String delimiter) throws IOException, ParseException {
		// Read File from csv
		//first column in csv has to be the dependent variable
		
		//initialize reader and variables
		BufferedReader br = new BufferedReader(new FileReader(trainSource));
		String line = null;
		int numRows=1;
		NumberFormat f = NumberFormat.getInstance(Locale.US); //define Number format in csv
		List<String> values= new LinkedList<String>();

		//read first line 
		line=br.readLine();
		String[] helper=line.split(delimiter);

		for(String str:helper){
			values.add(str);
		}

		//count columns
		int numColumns = StringUtils.countMatches(line, ";")+1;

		//read all lines to a String array
		while ((line = br.readLine()) != null) {	

			helper=line.split(delimiter);

			for(String str:helper){
				values.add(str);		
			}
		}
		//count Rows
		numRows=values.size()/numColumns;
		
		//change String Array to double Array
		double[] valueArray= new double[values.size()];

		for(int i=0;i<valueArray.length;i++){
			valueArray[i]= f.parse(values.get(i)).doubleValue();
		}
		
		//create Array to build covertree
		double[][] covertreeArray=new double[numRows][numColumns];

		for(int row=0; row<numRows;row++){
			for(int col=0;col<numColumns;col++){
				covertreeArray[row][col]=valueArray[(row*numColumns+col)];
			}
		}


		br.close();

		//buffered reader over
		System.out.println("Reading testdata done");
		
		
		//create Covertree
		CoverTree<Double> CT = new CoverTree<Double>();

		for(int i =0;i<covertreeArray.length;i++){
			CT.insert((double)i, covertreeArray[i]);	
		}
		
		System.out.println("Covertree built");

		//load test data from test file
		br= new BufferedReader(new FileReader(testSource));
		line = null;
		numRows=1;
		values= new LinkedList<String>();

		line=br.readLine();
		helper=line.split(delimiter);

		for(int i=0; i<helper.length;i++){
			values.add(helper[i]);
		}


		//count columns
		numColumns = StringUtils.countMatches(line, ";")+1;
		
		while ((line = br.readLine()) != null) {	
			numRows++;
			helper=line.split(delimiter);
			for(int i=0; i<helper.length;i++){
				values.add(helper[i]);
			}
		}

		//create Array to test
		double[][] testDataArray=new double[numRows][numColumns];
		
		valueArray= new double[values.size()];

		for(int i=0;i<valueArray.length;i++){
			valueArray[i]= f.parse(values.get(i)).doubleValue();
		}

		for(int row=0; row<numRows;row++){
			for(int col=0;col<numColumns;col++){

				testDataArray[row][col]=valueArray[(row*numColumns+col)];
			}
		}
		br.close();
		
		//testdata reader over
		System.out.println("Reading testdata over");
		
		//check timestamp
		java.util.Date date= new java.util.Date();
		long time= System.currentTimeMillis();
		


		//write result to csv
		BufferedWriter bw = new BufferedWriter(new FileWriter(testOutput));

		
		for(int i=0; i<numRows;i++){
			
			//transform data to required input format
			double[] val = new double[numColumns];
			for(int j=0;j<numColumns;j++){
				 val[j]=testDataArray[i][j];
			}
			//do forecast and write result to outputfile

			bw.write(Double.toString(CT.getForecastAllAIC(val))+System.getProperty("line.separator"));

		}
		bw.flush();
		bw.close();

		System.out.println("written");
		System.out.println("Forecast time: "+(System.currentTimeMillis()-time));

	}

}
