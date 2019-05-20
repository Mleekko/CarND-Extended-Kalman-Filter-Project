#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::sqrt;
using std::atan2;
using std::max;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init() {
  x_ = VectorXd(4);

  // FIXME:
  // State covariance matrix P
  P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1000, 0,
      0, 0, 0, 1000;

  // The initial transition matrix F_
  F_ = MatrixXd(4, 4);
  F_ << 1, 0, 1, 0,
      0, 1, 0, 1,
      0, 0, 1, 0,
      0, 0, 0, 1;

  // H_ =
  // R_ =
  Q_ = MatrixXd(4, 4);
  h_x = VectorXd(3);
  x_size = x_.size();
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;

  DoUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd y = CalculateRadarY(z);

  DoUpdate(y);
}

VectorXd KalmanFilter::CalculateRadarY(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // convert from cartesian coordinates to polar coordinates
  float rho = max(sqrt(px * px + py * py), 0.0001f);
  float phi = atan2(py, px);
  float rho_dot = (px * vx + py * vy) / rho;

  h_x << rho, phi, rho_dot;

  VectorXd y = z - h_x;

  // ensure that phi is in range [-pi, pi]
  while (y(1) > M_PI) {
    y(1) -= 2 * M_PI;
  }

  while (y(1) < -M_PI) {
    y(1) += 2 * M_PI;
  }

  return y;
}

void KalmanFilter::DoUpdate(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();

  // new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
