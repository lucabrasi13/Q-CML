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
// Source-File 
//

#include "grad_descent.h"

//
// Function definition
//

int numRow(string fname){
//Function to find the number of rows from the dataset
	ifstream inFile;
	inFile.open(fname);
	int numline = 0;
	string line;
	while(getline(inFile,line)){++numline;}

return numline;
}

int numCol(string fname){
//Function to find the number of cols from the dataset
    fstream myfile(fname);
    string line, temp;
    stringstream ss;
    int ncols=0;
    getline(myfile, line);
	ss << line;    
    while (ss)
        {
        ss >> temp;
        ncols++;
        }

return ncols-1;  
}

mat loadtext(string fname){
// Function to load the dataset into a matrix
	int r = numRow(fname);
	int c = numCol(fname);
	mat D(r,c),a(r,1);
	ifstream inFile;
	inFile.open(fname);
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){	
			inFile >> D(i,j);		
		}	
	}
	for(int i=0;i<r;i++){a(i,0)=1.;}

return concat_horizontal(a,D); /*First column is always ones*/
}

double cost_linear_regression(mat X, mat y, mat theta){
//Function to compute the cost function for Linear Regression
	it_assert(X.cols() == theta.rows() && X.rows() == y.rows(),"Incorrect Matrix dimensions"); //Error Handling

	int M = X.rows(); 
	double J = sumsum(pow(X*theta - y,2))/(2*M);

return J;
}

mat gradient_descent(mat X, mat y, mat theta, double alpha, int num_iters){
// Function to find the parameters theta using the gradient descent algorithm
	it_assert(X.cols() == theta.rows() && X.rows() == y.rows(),"Incorrect Matrix dimensions"); //Error Handling
	double J = 0.;	
	int M = X.rows();
	for(int i=0;i<num_iters;i++){
		J = cost_linear_regression(X,y,theta);
		theta = theta - ((alpha/M)*transpose((transpose(X*theta-y))*X)); 
	}

return theta;
}

mat visualize_cost_gradient_descent(mat X, mat y, mat theta, double alpha, int num_iters){
// Function to find the cost with each iteration of Gradient descent 
	it_assert(X.cols() == theta.rows() && X.rows() == y.rows(),"Incorrect Matrix dimensions"); //Error Handling
	mat J(num_iters,1);	
	int M = X.rows();
	for(int i=0;i<num_iters;i++){
		J(i,0) = cost_linear_regression(X,y,theta);
		theta = theta - ((alpha/M)* transpose((transpose(X*theta - y))*X)); 
	}
return J;
}

