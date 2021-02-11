#include <iostream>
#include <vector>
#include "math.h"

using namespace std;

void print_matrix(vector<vector<double>>& matrix){
    for(int i=0; i < matrix.size(); ++i){
        for(int j=0; j < matrix[0].size(); ++j){
            cout << matrix[i][j] << " ";
        }
        cout << "\n";
    }
    cout <<"\n";
}

//Uses gramn-schmidt to get the Q matrix for a matrix X with n rows and m columns
//O(max(n,m)^3) time complexity.
vector<vector<double>> get_Q(vector<vector<double>>& X, int n, int m){
    vector<vector<double>> Q(n, vector<double>(m, 0));
    for(int col=0; col < m; ++col){
        //We first project the col'th column of X onto the span of the first j-1 columns of Q:
        vector<double> projection(n, 0);
        for(int j=0; j < col; ++j){
            double inner_product = 0;
            for(int k=0; k < n; ++k){
                inner_product += Q[k][j]*X[k][col];
            }
            for(int k=0; k < n; ++k){
                projection[k] += inner_product*Q[k][j];
            }
        }

        //Now we compute the difference vector between the projection and the actual column vector
        //Note: total is used for storing the norm of the column
        double total = 0;
        for(int row=0; row < n; ++row){
            Q[row][col] = X[row][col] - projection[row];
            total += Q[row][col]*Q[row][col];
        }

        //Here we normalize the column added to Q
        for(int row=0; row < n; ++row){
            if(col == 0){
                cout << Q[row][col] << " " << sqrt(total) <<"\n\n";
            }
            Q[row][col] = Q[row][col] / sqrt(total);
        }
    }
    return Q;
}

//O(n^2) time complexity
vector<vector<double>> transpose(vector<vector<double>>& Q, int n, int m){
    vector<vector<double>> QT(m, vector<double>(n, 0));
    for(int i=0; i < m; ++i){
        for(int j=0; j < n; ++j){
            QT[i][j] = Q[j][i];
        }
    }
    return QT;
}

//Assumes A and B are nonempty matrices in O(n^3) time complexity. Assumes the # of columns of A
//is the same as the number of rows of B.
vector<vector<double>> mult(vector<vector<double>>& A, vector<vector<double>>& B){
    vector<vector<double>> answer(A.size(), vector<double>(B[0].size()));

    for(int row=0; row < A.size(); ++row){
        for(int col=0; col < B[0].size(); ++col){
            //inner product of A[row] with B[][col]
            double inner_product = 0;
            for(int i=0; i < A[0].size(); ++i){
                inner_product += A[row][i]*B[i][col];
            }
            answer[row][col] = inner_product;
        }
    }

    return answer;
}

//O(n^3) time complexity because of matrix multiplication (theoretically one can get it down to O(n^(2.3)) using Coppersmith-Winograd)
vector<vector<double>> get_R(vector<vector<double>>& Q, vector<vector<double>>& X, int n, int m){
    //R = Q^(T) X
    vector<vector<double>> QT = transpose(Q, n, m);
    return mult(QT, X);
}

//decomposes a matrix X into its QR decomposition. n is the number of rows and n is the number of columns.
pair<vector<vector<double>>, vector<vector<double>>> decompose(vector<vector<double>>& X, int n, int m){
    vector<vector<double>> Q = get_Q(X, n, m);
    vector<vector<double>> R = get_R(Q, X, n, m);
    return make_pair(Q, R);
}

vector<double> get_beta(vector<vector<double>>& R, vector<vector<double>> QT, vector<vector<double>>& y){
    //Rb = Q^(T)y := A

    vector<double> beta(R.size(), 0);
    vector<vector<double>> A = mult(QT, y);
    for(int row=y.size()-1; row >= 0; row--){
        if(row == R.size()-1){
            beta[row] = A[row][0] / R[row][row]; //note: A is a column vector
        }else{
            double prev_sum = 0;
            for(int col=y.size()-1; col > row; col--){
                prev_sum += R[row][col]*beta[col];
            }
            beta[row] = (A[row][0] - prev_sum) / R[row][row];
        }
    }

    return beta;
}

//Solves the least squares problem
int main(){

    //Test 1----------------------------------------------------------
    vector<vector<double>> X = {
        {2, 0, 0},
        {0, 3, 0},
        {0, 0, 4}
    };
    pair<vector<vector<double>>, vector<vector<double>>> test = decompose(X, 3, 3);

    cout << "Test Matrix 1: \n";
    print_matrix(X);
    cout << "Q: \n";
    print_matrix(test.first);
    cout << "R: \n";
    print_matrix(test.second);

    //Test 2----------------------------------------------------------
    X = {
        {2, 1, 7},
        {3.4, 3, 24.5},
        {1.23443, 2.543, 4}
    };

    test = decompose(X, 3, 3);

    cout << "Test Matrix 2: \n";
    print_matrix(X);
    cout << "Q: \n";
    print_matrix(test.first);
    cout << "R: \n";
    print_matrix(test.second);
    cout << "Product of Q with R: \n";
    vector<vector<double>> product = mult(test.first, test.second);
    print_matrix(product);


    vector<vector<double>> y = {{1},{2},{3}};
    vector<vector<double>> Q = test.first;
    vector<vector<double>> R = test.second;
    //Rb = Q^(T)y


    vector<double> beta = get_beta(R, transpose(Q, Q.size(), Q[0].size()), y);
    cout << "Beta:\n";
    for(int i=0; i < beta.size(); ++i){
        cout << beta[i] << " ";
    }

    //The true values are 0.247179, 1.22031, -.102096
    return 0;
}