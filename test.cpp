// Copyright <2018> <ROHIT KASHYAP>

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, 
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
// ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
// THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//
// Test file
//

#include "grad_descent.h"

using namespace std;
using namespace itpp;

void write_file(vec a, vec b, vec c){
// Function to write the vectors to a data file to plot	
	int N = size(a);
	ofstream data;
    data.open("data.txt");
    for(int i=0;i<N; i++){
        data << fixed << a[i] <<"    "<< fixed << b[i] << "    " << fixed << c[i] << endl;
    }
    data.close();
}

void gnuplot_load(string fn)
{
// Function to pipeline and load gnuplot
	string comand="gnuplot -persist\n";
	FILE *pipe= popen(comand.c_str() , "w");
	ostringstream oss;
	oss.str(""); oss.clear();
	oss<<"load '"<< fn << "'" << endl;
	fprintf(pipe, "%s",oss.str().c_str());
	fflush(pipe);
	int r=pclose(pipe);
	if(r<0)cout<<"# error in pclose(), while using unix pipes!"<<endl;
}

void plot()
{
// Function to plot from a data file
	ofstream fout;
	fout.open("fgnuplot");
	fout << "reset" << endl;
	fout << "set grid" << endl;
	fout << "set xtics" << endl;	
	fout <<	"#" << endl;
	fout << "plot 'data.txt' u 1:2 w p,\'data.txt' u 1:3 w l" << endl;
	fout << "#" << endl;
	fout << endl;
	fout.close();
	gnuplot_load("fgnuplot");
}

int main()
{
	string fname = "ex1data1.txt";								// File name
	mat D = loadtext(fname);									// Load the data matrix
	int r = D.rows();											// Find the row and column dimension
	int c = D.cols() -1; 					
	mat X = D(0,r-1,0,c-1);										// Separate the input and output label from the dataset
	mat y = D(0,r-1,c,c);			
	mat t = zeros(c,1);											// Empty vector to store the parameters theta
	double J = cost_linear_regression(X,y,t);					// Compute the cost function
	
	mat theta = gradient_descent(X,y,t,0.01,1500);				// Find the parameters theta
	mat ynew = X * theta;										// Map the input to the output
	write_file(D.get_col(1),D.get_col(2),ynew.get_col(0));		// Plot the result
	plot();
	
return 0;
}
