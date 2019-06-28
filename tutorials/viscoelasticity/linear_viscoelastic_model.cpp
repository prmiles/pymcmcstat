#include <cmath>
#include <iostream>

extern "C" double* linear_viscoelastic_model(float eta, float gamma, double *stretch, double *time, int nx){
double* q = new double[nx];
double tau = eta/gamma; // viscoelastic time constant
double dt;
double Tnc;
double Tpc;
q[0] = 0.0;
for (int ii = 0; ii < nx; ii++){
        dt = time[ii+1] - time[ii];
        Tnc = 1 - dt/(2*tau);
        Tpc = 1 + dt/(2*tau);
        q[ii+1] = pow(Tpc, -1)*(Tnc*q[ii] + gamma*(stretch[ii+1] - stretch[ii]));
    }
return q;
}