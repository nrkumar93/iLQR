#ifndef _DYNAMICS_H_
#define _DYNAMICS_H_

#include "common.h"
#include "config.h"
#include "mujoco_interface.h"

class Model {
public:
  virtual VectorXd dynamics(const VectorXd& x, const VectorXd& u) = 0;
  virtual VectorXd dynamics(const VectorXd& x, const VectorXd& u, bool step) {};
  virtual VecOfVecXd findInitU(VectorXd& x0, VectorXd& xF, int N, double dt) {};
  virtual double cost(const VectorXd& x, const VectorXd& u) = 0;
  virtual double final_cost(const VectorXd& x) = 0;

  virtual void setGoal(VectorXd& g) {};

  VectorXd integrate_dynamics(const VectorXd& x, const VectorXd& u, double dt) {

    VectorXd x1(x_dims);

    if (use_mujoco)
    {
      x1 = dynamics(x, u, true);
      return x1;
    }

    x1 = x + dynamics(x,u)*dt;
    return x1;
  }

  MujocoPsopt mj_handle;
  mjModel* mj_model;
  mjData* mj_data;

  bool use_mujoco = false;

  VectorXd u_min, u_max;

  int x_dims;
  int u_dims;
};

#endif
