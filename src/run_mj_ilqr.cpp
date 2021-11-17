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
 * \file   run_mj_ilqr.cpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   11/11/21
 */

#include "common.h"
#include "ilqr.h"
#include "mj_model.h"

//#define DOF 1
#define DOF 6
//#define DOF 4

/// UNI PENDULUM
//double qS[] = {0.0};
//double qSd[] = {0.0};
//double qG[] = {M_PI};
//double qGd[] = {0.0};

//// UR5
double qS[] = {-2.20325, 1.25752, -0.260757, -6.23505, 12.5839, 5.37701}; /// already on the table
double qSd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double qG[] = {-0.76138, 1.25278, -0.270137, -6.24006, 12.5849, 8.30729}; /// already on the table
double qGd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

//// multi_pendulum
//double qS[] = {0.0, 0.0, 0.0, 0.0}; /// already on the table
//double qSd[] = {0.0, 0.0, 0.0, 0.0};
//double qG[] = {0.0, M_PI, 0.0, 0.0}; /// already on the table
//double qGd[] = {0.0, 0.0, 0.0, 0.0};


int main() {

//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/alt_joint/alt_joint.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/pendulum.xml";
  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/ur5_exp.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/ur5l.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/pendulum.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/arm_gripper/arm_hand.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/alt_joint/multi_pendulum.xml";

  // Load mujoco model
  double dt = 0.05;
  Model* mj = new MujocoModel(mjcf_file, dt);

  for (int i=0; i<mj->mj_model->nq; ++i)
  {
    qS[i] = mj->mj_handle.normalize_angle(qS[i]);
    qG[i] = mj->mj_handle.normalize_angle(qG[i]);
  }

  /// ilqr stuff
  iLQR* ilqr;
  int T = 299; // right now, (T+1) has to be divisible by 10 - see derivatives.cpp. TODO remove this constraint

  // set start state
  VectorXd x0;
  x0.resize(2*DOF);
  for (int i=0; i<DOF; ++i)
  {
    x0(i) = qS[i];
    x0(DOF+i) = qSd[i];
  }

  // set goal state
  VectorXd goal(2*DOF);
  for (int i=0; i<DOF; ++i)
  {
    goal(i) = qG[i];
    goal(DOF+i) = qGd[i];
  }

  mj->setGoal(goal);

  // Define problem
  ilqr = new iLQR(mj, dt);

  // Make initialization for control sequence
  VecOfVecXd u0;
//  VectorXd u_init(DOF); u_init.setZero();
//  for (int i=0; i<T; i++) u0.push_back(u_init);
  u0 = mj->findInitU(x0, goal, T+1, dt);


  // Solve!
  cout << "Run iLQR!" << endl;
  auto start = std::chrono::system_clock::now();
  ilqr->generate_trajectory(x0, u0);
  auto now = std::chrono::system_clock::now();
  long int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  cout << "iLQR took: " << elapsed/1000. << " seconds." << endl;

  VecOfVecXd x_traj = ilqr->getFinalXTraj();
  VecOfVecXd u_traj = ilqr->getFinalUTraj();
  x_traj.pop_back();
  std::vector<double> t_traj;
  for (int i=0; i<T+1; ++i) t_traj.push_back(i*dt);

///// Call Visualizer
//  const char* mjcf_pb_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/alt_joint/alt_joint_pb.xml";
//  const char* mjcf_pb_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/pendulum_pb.xml";
  const char* mjcf_pb_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/ur5_exp_pb.xml";
//  const char* mjcf_pb_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/ur5l_pb.xml";
//  const char* mjcf_pb_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/alt_joint/multi_pendulum_pb.xml";
  mj->mj_handle.setupFromMJCFFile(mjcf_pb_file);

  std::vector<std::vector<double>> x_vec, dx_vec, u_vec;
  for (int i=0; i<x_traj.size(); ++i)
  {
    std::vector<double> unit_x(x_traj[i].data(), x_traj[i].data()+DOF);
    std::vector<double> unit_dx(x_traj[i].data()+DOF, x_traj[i].data()+2*DOF);
    std::vector<double> unit_u(u_traj[i].data(), u_traj[i].data()+DOF);

    x_vec.push_back(unit_x);
    dx_vec.push_back(unit_dx);
    u_vec.push_back(unit_u);
  }

  mj->mj_handle.visualize(t_traj.back(), t_traj, x_vec, dx_vec);


  return 0;
}
