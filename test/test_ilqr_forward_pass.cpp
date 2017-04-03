#include "gtest/gtest.h"
#include "ilqr.h"
#include "double_integrator.h"
#include "eigen_helpers.h"

static const double eq_tol = 1e-3;

class ILQRSetup : public ::testing::Test
{
public:
  virtual void SetUp()
  {
    double dt = 0.05;
    int T = 50;
    VectorXd goal(4);
    goal << 1.0, 1.0, 0.0, 0.0;
    DoubleIntegrator* simple_model = new DoubleIntegrator(goal);

    ilqr_simple.reset(new iLQR(simple_model, dt, T));

    x0.resize(4);
    xd.resize(4);

    // Previous (easier) test case
    // x0 << 0., 0., 0., 0.;
    // Vector2d u_init(0.1, 0.1);
    // expected_1 << 0., 0., 0.005, 0.005;
    // expected_end << 0.30625, 0.30625, 0.25, 0.25;

    x0 << 0.1, -0.5, 0.1, 1.;
    Vector2d u_init(-0.1, 0.1);

    for (int i=0; i<T; i++)  u0.push_back(u_init);

    init_cost = ilqr_simple->init_traj(x0, u0);
  }

  VectorXd x0, xd;
  VecOfVecXd u0;
  double init_cost;

  std::shared_ptr<iLQR> ilqr_simple;
};

// Just testing initial rollout
TEST_F(ILQRSetup, ForwardPassTest)
{
  Eigen::VectorXd expected_1(4), expected_end(4);
  expected_1 << 0.105, -0.45, 0.095, 1.005;
  expected_end << 0.0437, 2.3062, -0.15, 1.25;

  Eigen::VectorXd x1 = ilqr_simple->xs[1];
  Eigen::VectorXd end = ilqr_simple->xs[50];
  // std::cout << "-\n" << x1 << std::endl;
  // std::cout << "-\n" << end << std::endl;

  EXPECT_TRUE(ilqr_simple->xs[1].isApprox(expected_1, eq_tol));
  EXPECT_TRUE(end.isApprox(expected_end, eq_tol));
  // TODO test cost
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
