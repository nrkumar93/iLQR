/*
 * Copyright (c) 2021, Ramkumar Natarajan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Carnegie Mellon University nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/*!
 * \file   mj_model.h
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   11/11/21
 */

#ifndef ILQR_MJ_MODEL_H
#define ILQR_MJ_MODEL_H

#include "model.h"

/*
  Mujoco model API for iLQR
  state = [x1, x2, ... , xN, dx1, dx2, ... , dxN]
  control = [u1, u2, ..., uM]
  dx = [dx1, dx2, ... , dxN, ddx1, ddx2, ... , ddxN]
*/
class MujocoModel : public Model {

public:
  MujocoModel(const char* mjcf_file, double dt)
  {

    mj_handle.setupFromMJCFFile(mjcf_file);
    mj_model = mj_handle.getMjModel();
    mj_data = mj_handle.getMjData();

    mj_model->opt.timestep = dt;

    use_mujoco = true;

    x_dims = mj_model->nq + mj_model->nv;
    u_dims = mj_model->nu;
    u_min.resize(u_dims);
    u_max.resize(u_dims);

    /// bounds on u
    for (int i=0; i<u_dims; ++i)
    {
      u_min(i) = -100;
      u_max(i) = 100;
    }
  }

  void setGoal(VectorXd& g)
  {
    goal = g;
//    goal.resize(mj_model->nq);
//    for (int i=0; i<mj_model->nq; ++i)
//    {
//      goal(i) = g[i];
//    }
  }

  virtual VectorXd dynamics(const VectorXd& x, const VectorXd& u) override
  {
    dynamics(x, u, false);
  }

  virtual VectorXd dynamics(const VectorXd& x, const VectorXd& u, bool step) override {

    VectorXd dx(x_dims);

    const double* q = x.head(mj_model->nq).data();
    const double* dq = x.tail(mj_model->nv).data();
    const double* tau = u.head(mj_model->nu).data();
    double ddq[mj_model->nv];

    if (step)
    {
      mj_handle.step(q, dq, tau, ddq);
      for (int i=0; i<mj_model->nq; ++i)
      {
        dx(i) = mj_handle.normalize_angle(mj_data->qpos[i]);
      }
      for (int i=mj_model->nq, j=0; i<x_dims && j<mj_model->nv; ++i, ++j)
      {
        dx(i) = mj_data->qvel[j];
      }
    }
    else
    {
      mj_handle.forwardSimulate(q, dq, tau, ddq);
      dx.head(mj_model->nq) = x.tail(mj_model->nq);
      for (int i=mj_model->nq, j=0; i<x_dims && j<mj_model->nv; ++i, ++j)
      {
        dx(i) = ddq[j];
      }
    }

    return dx;
  }

  virtual double cost(const VectorXd& x, const VectorXd& u) override {

    bool incl_frc_diff = true;
    bool incl_ctrl = false;
    bool incl_jnt_dist = false;

    double frc_diff_cost = 0;
    double ctrl_cost = 0;
    double jnt_dist_cost = 0;

    double w_frc_diff = 1;
    double w_ctrl = 1;
    double w_jnt_dist = 1;

    double total_cost = 0;

    /// Net torque sensed (qfrc_constraint - qfrc_unc)
    if (incl_frc_diff)
    {
      dynamics(x, u, false);

      double qfrc_constraint[mj_model->nv], qfrc_unc[mj_model->nv];
      mju_copy(qfrc_constraint, mj_data->qfrc_constraint, mj_model->nv);
      mju_copy(qfrc_unc, mj_data->qfrc_unc, mj_model->nv);

      // force diff
      double frc_diff[mj_model->nv];
      for (int i=0; i<mj_model->nv; ++i)
      {
        int unc_sign = qfrc_unc[i]<0?-1:1;
        int con_sign = qfrc_constraint[i]<0?-1:1;

        if (unc_sign != con_sign)
        {
          if (abs(qfrc_constraint[i]) >= abs(qfrc_unc[i]))
          {
            frc_diff[i] = 0.0;
          }
          else
          {
            frc_diff[i] = qfrc_constraint[i] + qfrc_unc[i];
          }
        }
        else
        {
          frc_diff[i] = qfrc_constraint[i] + qfrc_unc[i];
        }
      }

      for (int i=0; i<mj_model->nq; ++i)
      {
        frc_diff_cost += frc_diff[i] * frc_diff[i];
      }
    }

    /// Control cost. Norm of torque at actuation
    if (incl_ctrl)
    {
      ctrl_cost = u.dot(u);

      total_cost += ctrl_cost;
    }

    /// Cspace distance
    if (incl_jnt_dist)
    {
      for (int i=0; i<mj_model->nq; ++i)
      {
        jnt_dist_cost += std::pow(mj_handle.shortest_angular_distance(x(i),goal(i)), 2);
      }

      total_cost += jnt_dist_cost;
    }

    total_cost = w_frc_diff*frc_diff_cost + w_ctrl*ctrl_cost + w_jnt_dist*jnt_dist_cost;

//    // Printing
////  std::cout << "frc_diff_norm: " << frc_diff_norm << " del_q_norm: " << del_q_norm << std::endl;
//  std::cout << "frc_diff_cost: " << frc_diff_cost << "\t" << "ctrl_cost: " << ctrl_cost << "\t" << "jnt_dist_cost: " << jnt_dist_cost << "\n";

    std::cout << "=================================" << std::endl;
    std::cout << "x\n----------\n" << x << std::endl;
    std::cout << "u\n----------\n" << u << std::endl;
    std::cout << "total_cost: " << total_cost << std::endl;
    std::cout << "=================================" << std::endl;

    return total_cost;
  }

  virtual double final_cost(const VectorXd& x) override {
    double rho=100;
    double jnt_dist_cost = 0;
    for (int i=0; i<mj_model->nq; ++i)
    {
      jnt_dist_cost += std::pow(mj_handle.shortest_angular_distance(x(i),goal(i)), 2);
    }
    return rho*jnt_dist_cost;

//    double cost = (goal-x).transpose()*(goal-x);
//    return rho*cost;
  }

  virtual VecOfVecXd findInitU(VectorXd& x0, VectorXd& xF, int N, double dt)
  {
    VecOfVecXd u_init;
    u_init.reserve(N);

    double prev_q[mj_model->nq], prev_dq[mj_model->nv];
    for (int i=0; i<mj_model->nq; ++i)
    {
      prev_q[i] = x0(i);
      prev_dq[i] = x0(mj_model->nq+i);
    }

    for (int i=0; i<N; ++i)
    {
      double q[mj_model->nq];
      double dq[mj_model->nv];
      double ddq[mj_model->nv];
      mju_zero(ddq, mj_model->nv);

      const double* q0 = x0.head(mj_model->nq).data();
      const double* dq0 = x0.tail(mj_model->nv).data();
      const double* qF = xF.head(mj_model->nq).data();
      const double* dqF = xF.tail(mj_model->nv).data();

      for (int j=0; j<mj_model->nq; ++j)
      {
        q[j] = mj_handle.interpolateAngle(q0[j], qF[j], j*(1.0/N));
      }
      for (int j=0; j<mj_model->nv; ++j)
      {
        dq[j] = mj_handle.interp1(0, N-1, i, dq0[j], dqF[j]);
//        dq[j] = mj_handle.shortest_angular_distance(prev_q[j],q[j])/dt;
//        ddq[j] = (dq[j]-prev_dq[j])/dt;
      }

      double tau[mj_model->nu];
      mj_handle.inverseSimulate(q, dq, ddq, tau);

      VectorXd unit_u_init(mj_model->nu);
      for (int j=0; j<mj_model->nu; ++j)
      {
        unit_u_init(j) = tau[j];
      }
      u_init.push_back(unit_u_init);

      for (int i=0; i<mj_model->nq; ++i)
      {
        prev_q[i] = q[i];
        prev_dq[i] = dq[i];
      }
    }
    return u_init;
  }

private:
  VectorXd goal;
};

#endif //ILQR_MJ_MODEL_H

