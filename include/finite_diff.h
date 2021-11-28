#ifndef _FINITE_DIFF_H_
#define _FINITE_DIFF_H_

#include "common.h"
#include "eigen/Eigen/Core"
#include <iostream>
#include <cmath>

static const double eps = 1e-3;
//static const double eps = 1;

using Eigen::VectorXd;
using Eigen::MatrixXd;


double finite_diff_gradient(std::function<double(double)> f, double x) {
  double plus = x+eps;
  double minus = x-eps;
  return (f(plus)-f(minus)) / (2*eps);
}

// TODO pass in reference to output
VectorXd finite_diff_gradient(std::function<double(VectorXd)> f, VectorXd x) {
  int n_dims = x.size();
  VectorXd plus(n_dims), minus(n_dims), dx(n_dims);

  for (int i=0; i<n_dims; i++) {
    plus = minus = x;
    plus[i] += eps;
    minus[i] -= eps;
    dx[i] = (f(plus)-f(minus)) / (2*eps);
  }
  return dx;
}

MatrixXd finite_diff_jacobian(std::function<VectorXd(VectorXd)> f, VectorXd x, int out_size) {
  int n_dims = x.size();
  VectorXd plus(n_dims), minus(n_dims);
  MatrixXd dx(out_size, n_dims);

  for (int i=0; i<n_dims; i++) {
    plus = minus = x;
    plus(i) += eps;
    minus(i) -= eps;
    dx.col(i) = (f(plus)-f(minus)) / (2*eps);
  }
  return dx;
}

void finite_diff_vecvec2scalar(std::function<double(VectorXd, VectorXd)> f, 
    const VectorXd& x1, const VectorXd& x2, MatrixXd& out) {
  // assume out is already right size?
  VectorXd p1, p2, m1, m2;

  for (int i=0; i<x1.size(); i++){
    for (int j=0; j<x2.size(); j++){  
        p1 = m1 = x1;
        p2 = m2 = x2;
        p1[i] += eps;
        m1[i] -= eps;
        p2[j] += eps;
        m2[j] -= eps;
        out(i,j) = (f(p1, p2) - f(m1, p2) - f(p1, m2) + f(m1, m2)) / (4*eps*eps);
    }
  }
}

void finite_diff_hessian(std::function<double(VectorXd)> f, const VectorXd& x, MatrixXd& out) {
  VectorXd pp, pm, mp, mm; //plus-plus, plus-minus, ....

  int n = x.size();

  for (int i=0; i<n; i++) {
    for (int j=i; j<n; j++) {
      pp = pm = mp = mm = x;
      pp[i] += eps;
      pp[j] += eps;
      pm[i] += eps;
      pm[j] -= eps;
      mp[i] -= eps;
      mp[j] += eps;
      mm[i] -= eps;
      mm[j] -= eps;
      out(i,j) = out(j,i) = (f(pp) - f(mp) - f(pm) + f(mm)) / (4*eps*eps);
    }
  }
}

#endif
