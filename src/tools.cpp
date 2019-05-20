#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::invalid_argument;
using std::logic_error;
using std::to_string;

Tools::Tools() {
  rmse = VectorXd(4);
  Hj = MatrixXd(3, 4);
}

Tools::~Tools() {
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  rmse << 0, 0, 0, 0;

  if (estimations.empty()) {
    throw invalid_argument("Estimations should not be empty!");
  }

  if (estimations.size() != ground_truth.size()) {
    throw invalid_argument(
        "The estimations and ground truth vectors should be of same size! "
            + to_string(estimations.size()) + " vs " + to_string(ground_truth.size()));
  }

  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  // recover state parameters
  float p_x = x_state(0);
  float p_y = x_state(1);
  float v_x = x_state(2);
  float v_y = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  float c1 = p_x * p_x + p_y * p_y;
  float c2 = std::sqrt(c1);
  float c3 = (c1 * c2);

  float vp = v_x * p_y;
  float pv = v_y * p_x;
  float vp_pv_c3 = (vp - pv) / c3;
  float p_x_c2 = p_x / c2;
  float p_y_c2 = p_y / c2;

  // prevent division by zero
  if (std::fabs(c1) < 0.0001) {
    throw logic_error("Error - Division by Zero");
  }

  // compute the Jacobian matrix
  Hj << p_x_c2, p_y_c2, 0, 0,
      -(p_y / c1), (p_x / c1), 0, 0,
      p_y * vp_pv_c3, p_x * -vp_pv_c3, p_x_c2, p_y_c2;

  return Hj;
}
