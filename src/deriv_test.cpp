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
 * \file   deriv_test.cpp
 * \author Ramkumar Natarajan (rnataraj@cs.cmu.edu)
 * \date   11/19/21
 */

#include "common.h"
#include "ilqr.h"
#include "mj_model.h"

//#define DOF 1
//#define DOF 6
//#define DOF 4
#define DOF 2

/// UNI PENDULUM
//double qS[] = {0.0};
//double qSd[] = {0.0};
//double qG[] = {M_PI};
//double qGd[] = {0.0};

//// UR5
//double qS[] = {-2.20325, 1.25752, -0.260757, -6.23505, 12.5839, 5.37701}; /// already on the table
//double qS[] = {-2.20325, 1.25752, -0.260757, -6.23505}; /// already on the table
//double qSd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//double qG[] = {-0.76138, 1.25278, -0.270137, -6.24006, 12.5849, 8.30729}; /// already on the table
//double qG[] = {-0.76138, 1.25278, -0.270137, -6.24006}; /// already on the table
//double qG[] = {-2.19325, 1.25752, -0.260757, -6.23505}; /// already on the table
//double qGd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double qS[] = {-2.55714, 1.25741}; /// already on the table
double qG[] = {-2.35714, 1.25741}; /// already on the table
double qSd[] = {0.0, 0.0};
double qGd[] = {0.0, 0.0};

//// multi_pendulum
//double qS[] = {0.0, 0.0, 0.0, 0.0}; /// already on the table
//double qSd[] = {0.0, 0.0, 0.0, 0.0};
//double qG[] = {0.0, M_PI, 0.0, 0.0}; /// already on the table
//double qGd[] = {0.0, 0.0, 0.0, 0.0};
//double qS[] = {0.0, 0.0}; /// already on the table
//double qSd[] = {0.0, 0.0};
//double qG[] = {0.0, M_PI}; /// already on the table
//double qGd[] = {0.0, 0.0};

int main()
{

//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/alt_joint/alt_joint.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/pendulum.xml";
  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/ur5_exp.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/ur5l.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/pendulum.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/arm_gripper/arm_hand.xml";
//  const char* mjcf_file = "/home/gaussian/cmu_ri_phd/phd_misc/mujoco210/model/alt_joint/multi_pendulum.xml";

  // Load mujoco model
  double dt = 0.002;
  Model* mj = new MujocoModel(mjcf_file, dt);

  for (int i=0; i<mj->mj_model->nq; ++i)
  {
    qS[i] = mj->mj_handle.normalize_angle(qS[i]);
    qG[i] = mj->mj_handle.normalize_angle(qG[i]);
  }

  iLQR* ilqr;
  ilqr = new iLQR(mj, dt);

  ilqr->T = 1;

  // set start state
  VectorXd x0;
  x0.resize(2*DOF);
  for (int i=0; i<DOF; ++i)
  {
    x0(i) = qS[i];
    x0(DOF+i) = qSd[i];
  }
  VecOfVecXd x;
  x.push_back(x0);

  // set goal state
  VectorXd goal(2*DOF);
  for (int i=0; i<DOF; ++i)
  {
    goal(i) = qG[i];
    goal(DOF+i) = qGd[i];
  }
  x.push_back(goal);
  mj->setGoal(goal);

  // Make initialization for control sequence
  VecOfVecXd tau;
  VectorXd u0(DOF); u0.setZero();
//  tau.push_back(u0);
  for (int i=0; i<ilqr->T; i++) tau.push_back(u0);

  VecOfMatXd fx, fu;
  VecOfVecXd cx; //nx(T+1)
  VecOfVecXd cu; //mx(T+1)
  VecOfMatXd cxx; //nxnx(T+1)
  VecOfMatXd cxu; //nxmx(T+1)
  VecOfMatXd cuu; //mxmx(T+1)

  fx.resize(ilqr->T);
  fu.resize(ilqr->T);
  cx.resize(ilqr->T+1);
  cu.resize(ilqr->T+1);
  cxu.resize(ilqr->T+1);
  cxx.resize(ilqr->T+1);
  cuu.resize(ilqr->T+1);

  ilqr->get_dynamics_derivatives(x, tau, fx, fu);
  ilqr->get_cost_derivatives(x, tau, cx, cu);
  ilqr->get_cost_2nd_derivatives(x, tau, cxx, cxu, cuu);

  std::cout << "fx:\n" << fx[0] << std::endl;
  std::cout << "fu:\n" << fu[0] << std::endl;
  std::cout << "cx:\n" << cx[0] << std::endl;
  std::cout << "cu:\n" << cu[0] << std::endl;
  std::cout << "cxx:\n" << cxx[0] << std::endl;
  std::cout << "cxu:\n" << cxu[0] << std::endl;
  std::cout << "cuu:\n" << cuu[0] << std::endl;

}


