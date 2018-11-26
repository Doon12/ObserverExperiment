/*
 * Copyright 2011-2013 Mario Mulansky
 * Copyright 2012 Karsten Ahnert
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 * 
 * Original code is lorenz_ublas.cpp which is one of ODEINT library example
 * Oct 16 2018 Modified by JUNHAN LEE, proposed observer.cpp
 */

#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <ctime>
#include <vector>
#include <fstream>
#include <math.h>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint.hpp>
#include <Eigen/Dense>

#include <boost/assign/std/vector.hpp> 
#include <boost/random.hpp>
#include <boost/assert.hpp>
#include <unsupported/Eigen/MatrixFunctions>

// Include this header file to get access to VectorNav sensors.
#include "vn/sensors.h"
// We need this header file for our sleep function.
#include "vn/thread.h"

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

using namespace Eigen;
using namespace std;
using namespace boost::numeric::odeint;
typedef std::vector<double, Eigen::aligned_allocator<double>> vec_d;
enum observer_mode{ Proposed, Mahony }; 
static observer_mode myobserver = Mahony;
static double ki = 1.5;
static double kp = 2.5;
static Matrix3d W;
static Matrix<double,3,1> b;
static Matrix<double,3,1> e1;
static Matrix<double,3,1> e2;
static Matrix<double,3,1> e3;
static Matrix3d S;
static Matrix3d C;
static Matrix3d F;



Matrix3d hat_map(Matrix<double, 3, 1> e){
    Matrix3d hat_e;
    hat_e(0,0) = 0;
    hat_e(0,1) = -e(2,0);
    hat_e(0,2) = e(1,0);
    hat_e(1,0) = e(2,0);
    hat_e(1,1) = 0;
    hat_e(1,2) = -e(0,0);
    hat_e(2,0) = -e(1,0);
    hat_e(2,1) = e(0,0);
    hat_e(2,2) = 0;

    return hat_e;
}

Matrix<double, 3, 1> Omega(const double t){
    MatrixXd Omega_t(3,1);
    Omega_t(0,0) = 1+cos(t);
    Omega_t(1,0) = sin(t)-sin(t)*cos(t);
    Omega_t(2,0) = cos(t)+sin(t)*sin(t);

    return Omega_t;
}


//[observers
void proposed_observer( const vec_d &x , vec_d &dxdt , const double t )
{
    Matrix3d dAbardt;
    Matrix<double, 3, 1> dbbardt;

    // Abar = reshape(y(1:9,:), [3,3]);
    // bbar = reshape(y(10:12,:), [3,1]);
    vec_d::const_iterator mid = x.begin() + 9;
    vec_d Abar_vec(x.begin(), mid);
    vec_d bbar_vec(mid, x.begin()+12);
    double *vA = &Abar_vec[0];
    Eigen::Map<Eigen::MatrixXd> Abar(vA,3,3);
    double *vb = &bbar_vec[0];
    Eigen::Map<Eigen::MatrixXd> bbar(vb,3,1);


    //A_t = S * W * C_t';
    Matrix3d A = S * W * C.transpose();

    //assign dAbardt
    dAbardt = Abar * hat_map(Omega) - A * hat_map(bbar) + kp * (A - Abar);

    //for n = 1:3
    //    dbbardt = dbbardt - ki * W(n,n) * cross(C_t(:,n), Abar' * S(:,n)); 
    //end
    dbbardt << 0.0, 0.0, 0.0;
    for (int i = 0 ; i < 3 ; i++){
        //(C_t.col(i)).cross(Abar.transpose() * (S.col(i)));
        Vector3d vec1;
        vec1 = C.col(i);
        Vector3d vec2;
        vec2 = Abar.transpose() * S.col(i);
        Vector3d vec3;
        vec3 = vec1.cross(vec2);
        dbbardt -= ki * W(i,i) * vec3; 
    }

    // save dxdt
    for(int i = 0 ; i < 12 ; i ++){
        if(i < 9){
            dxdt[i] = dAbardt(i%3, i/3);
        }
        else{
            dxdt[i] = dbbardt(i-9,0);
        }
    }
}

void mahony_observer( const vec_d &x , vec_d &dxdt , const double t )
{
    Matrix3d dRbardt;
    Matrix<double, 3, 1> dbbardt;

    // Rbar = reshape(y(1:9,:), [3,3]);
    // bbar = reshape(y(10:12,:), [3,1]);
    vec_d::const_iterator mid = x.begin() + 9;
    vec_d Rbar_vec(x.begin(), mid);
    vec_d bbar_vec(mid, x.begin()+12);
    double *vR = &Rbar_vec[0];
    Eigen::Map<Eigen::MatrixXd> Rbar(vR,3,3);
    double *vb = &bbar_vec[0];
    Eigen::Map<Eigen::MatrixXd> bbar(vb,3,1);

    //R_t = R(t);
    Matrix3d R_t = R(t);

    //V = R_t' * S;
    Matrix3d V = R_t.transpose() * S;

    //Vbar = Rbar' * S;
    Matrix3d Vbar =  Rbar.transpose() * S;

    //wmes += W(n,n) * cross(V(:,n), Vbar(:,n))
    Matrix<double,3,1> wmes;
    wmes << 0,0,0;
    for(int i = 0 ; i < 3; i ++){
        wmes += W(i,i) *  V.col(i).cross(Vbar.col(i));
    }

    //assign dRbardt
    dRbardt = Rbar * (hat_map(Omega(t) + b - bbar) + kp * hat_map(wmes));
    //assign dbbardt
    dbbardt = - ki * wmes;

    // save dxdt
    for(int i = 0 ; i < 12 ; i ++){
        if(i < 9){
            dxdt[i] = dRbardt(i%3, i/3);
        }
        else{
            dxdt[i] = dbbardt(i-9,0);
        }
    }
}
//]

//[print function
struct streaming_observer
{
    std::ostream &m_out;
    streaming_observer( std::ostream &out) : m_out(out){}
    ~streaming_observer() {}

    void operator()( const vec_d &x, double t) const
    {
        MatrixXd R_t = R(t);
        if(myobserver == Proposed){
            //R
            vec_d::const_iterator mid = x.begin() + 9;
            vec_d Abar_vec(x.begin(), mid);
            Map<MatrixXd> Abar_mat(Abar_vec.data(),3,3);
            MatrixXd Rbar_mat = F.inverse() * Abar_mat;
            //b
            vec_d bbar_vec(mid, x.begin()+12);
            Map<MatrixXd> bbar_mat(bbar_vec.data(),3,1);
            double err_norm_R = (Rbar_mat - R_t).norm();
            double err_norm_b = (bbar_mat - b).norm();
            //calculate error norm
            m_out  << t <<  "\t" << err_norm_R << "\t" << err_norm_b << "\n";
        }
        else{//Mahony
            //R            
            vec_d::const_iterator mid = x.begin() + 9;
            vec_d Rbar_vec(x.begin(), mid);
            Map<MatrixXd> Rbar_mat(Rbar_vec.data(),3,3);
            //b
            vec_d bbar_vec(mid, x.begin()+12);
            Map<MatrixXd> bbar_mat(bbar_vec.data(),3,1);

            double err_norm_R = (Rbar_mat - R_t).norm();
            double err_norm_b = (bbar_mat - b).norm();
            //calculate error norm
            m_out  << t <<  "\t" << err_norm_R << "\t" << err_norm_b << "\n";
        }
    }
};
//]

//[main
int main(int argc, char* argv[])
{
	//[camera settings
	int wait_time = 20;
	float actual_marker_length = 0.14; //this should be in meters
	cv::Mat image, image_copy;
	cv::Mat camera_matrix, dist_coeffs;
	std::ostringstream vector_to_marker
	//]

    //[var definition
    W << 1.0/3, 0.0, 0.0, 0.0, 1.0/3, 0.0, 0.0, 0.0, 1.0/3;
    //calc b by measuring difference between two gyros
	b << 1.0, 0.5, -1.0;

    e1 << 1.0, 0.0, 0.0;
    e2 << 0.0, 1.0, 0.0;
    e3 << 0.0, 0.0, 1.0;

    Matrix<double,3,1> s1;
    Matrix<double,3,1> s2;
    Matrix<double,3,1> s3;
    Matrix<double,3,1> c1;
    Matrix<double,3,1> c2;
    Matrix<double,3,1> c3;
 	//marker id 100
    s1 = 2.21733*e1 + 0.252446*e2 + 0.196596*e3;
	s1 /= s1.norm();
    //marker id 101
	s2 = 2.07142*e1 + 0.269657*e2 + 0.615426*e3;
	s2 /= s2.norm();
	//gravity direction
    s3 = 0.663*e1 + 9.774*e2 + 0.176*e3;
	s3 /= s3.norm();

    S << s1,s2,s3;
    std::cout << "S is " << S << std::endl;
    std::cout << "W is " << W << std::endl;    
    F = S * W * S.transpose();
    std::cout << "F is " << F << std::endl;

    vec_d bbar_init {0.0, 0.0, 0.0 };
    MatrixXd mat_Rbar_init(3,3);
    mat_Rbar_init << 0.2440,0.9107,-0.3333,
                    0.9107,-0.3333,-0.2440,
                    -0.3333,-0.2440,-0.9107;
    //]

	//[VectorNav Connect
	const string SensorPort = "/dev/ttyUSB0"; // LINUX
	const uint32_t SensorBaudrate = 115200;

	VnSensor vs;
	vs.connect(SensorPort, SensorBaudrate);
	
	// Let's query the sensor's model number.
	string mn = vs.readModelNumber();
	cout << "Model Number: " << mn << endl;

	//]

	//[open camera
	cv::VideoCapture in_video;
	in_video.open(1);
	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
	cv::FileStorage fs("../../calibration_params.yml", cv::FileStorage::READ);

	fs["camera_matrix"] >> camera_matrix;
	fs["distortion_coefficients"] >> dist_coeffs;

	std::cout << "camera matrix\n" << camera_matrix << std::endl;
	std::cout << "\ndist coeffs\n" << dist_coeffs << std::endl;
	//]
	while (in_video.grab()){
		in_video.retrieve(image);
		image.copyTo(image_copy);
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> corners;
		cv::aruco::detectMarkers(image, dictionary, corners, ids);

		//if at least one marker detected
		if(ids.size() > 0){
			cv::aruco::drawDetectedMarkers(image_copy, corners, ids);
			std::vector<cv::Vec3d> tvecs;
			cv::aruco::estimatePoseSingleMarkers(corners, actual_marker_length, camera_matrix, dist_coeffs, rvecs, tvecs);
			for(int i = 0 ; i < ids.size() ; i ++){
				// c1
				if(ids[i] == 100){
					c1 = tvecs[i];
				}
				// c2
				else if(ids[i] == 101){
					c2 = tvecs[i];
				}
				else{
					//do nothing
				}
			}
		}
		//calc acc
		c3 = vs.readAccelerationMeasurements();
		C << c1, c2, c3;

		//quit if ESC pressed
        cv::imshow("Pose estimation", image_copy);
        char key = (char) cv::waitKey(wait_time);
        if (key == 27)
            break;

	}

    //[observer simulation
    vec_d Y_init;
    Y_init.reserve(12);
    if(myobserver == Proposed){
        MatrixXd mat_Abar_init(3,3);
        mat_Abar_init = F * mat_Rbar_init;
        vec_d Abar_init(mat_Abar_init.data(), mat_Abar_init.data() + mat_Abar_init.size());
        Y_init.insert( Y_init.end(), Abar_init.begin(), Abar_init.end() );
    }
    else{
        vec_d Rbar_init(mat_Rbar_init.data(), mat_Rbar_init.data() + mat_Rbar_init.size());
        Y_init.insert( Y_init.end(), Rbar_init.begin(), Rbar_init.end() );
    }
    Y_init.insert( Y_init.end(), bbar_init.begin(), bbar_init.end() );
    for(int i = 0 ; i < Y_init.size() ; i ++){
        cout << Y_init[i] << endl;
    }
    
	// Experiment
    runge_kutta_dopri5<vec_d> stepper;
    const double dt = 0.1;
    double t = 0.0;
    for ( size_t i(0); i <= 100; ++i){
        stepper.do_step(proposed_observer, Y_init , t, dt );
        t += dt;

        streaming(Y_init, );
    }
    return 0;
}
//]
