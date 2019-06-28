#include <cmath>
#include <iostream>

extern "C" double* linear_model(float slope, float offset, double *x, int nx){
double* y = new double[nx];
for (int ii = 0; ii < nx; ii++){
        y[ii] = slope*x[ii] + offset;
    }
return y;
}