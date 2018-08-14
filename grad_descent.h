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
// grad_descent.h (Header file)
//
#include<iostream>
#include<iomanip>
#include<itpp/itbase.h>

using namespace std;
using namespace itpp;

//
// Declaration of functions
//

/*Function to find the number of rows from the dataset*/
int numRow(string fname);
/*Function to find the number of cols from the dataset*/
int numCol(string fname);
/*Function to load the dataset into a matrix*/
mat loadtext(string fname);
/*Function to compute the cost function for Linear Regression*/
double cost_linear_regression(mat X, mat y, mat theta);
/*Function to find the parameters theta using the gradient descent algorithm*/
mat gradient_descent(mat X, mat y, mat theta, double alpha, int num_iters);
/*Function to find the cost with each iteration of Gradient descent*/
mat visualize_cost_gradient_descent(mat X, mat y, mat theta, double alpha, int num_iters);

