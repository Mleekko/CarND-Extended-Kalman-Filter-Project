#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::cout;
using std::endl;
using std::vector;
using std::sin;
using std::cos;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
      0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  // 4D state to a 2D observation space conversion matrix - laser
  H_laser_ << 1, 0, 0, 0,
      0, 1, 0, 0;

  // acceleration noise components
  noise_ax = 9;
  noise_ay = 9;

}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    Initialize(measurement_pack);

    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the process noise covariance matrix.
   */

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  float dt_3_2 = dt_3 / 2;
  float dt_4_4 = dt_4 / 4;

  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // set the process covariance matrix Q
  ekf_.Q_ << dt_4_4 * noise_ax, 0, dt_3_2 * noise_ax, 0,
      0, dt_4_4 * noise_ay, 0, dt_3_2 * noise_ay,
      dt_3_2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3_2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

void FusionEKF::Initialize(const MeasurementPackage &measurement_pack) {
  // first measurement
  cout << "EKF: " << endl;
  ekf_.Init();

  previous_timestamp_ = measurement_pack.timestamp_;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    float rho = measurement_pack.raw_measurements_[0];
    float phi = measurement_pack.raw_measurements_[1];

    // use trigonometry to convert radar data from polar to cartesian coordinates.
    // skip rho_dot as we don't have enough data to convert from radial velocity to the real velocity
    ekf_.x_ << rho * cos(phi),
        rho * sin(phi),
        0,
        0;
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
    // set the state with the initial location and zero velocity
    ekf_.x_ << measurement_pack.raw_measurements_[0],
        measurement_pack.raw_measurements_[1],
        0,
        0;
  }

}