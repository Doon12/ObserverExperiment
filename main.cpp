#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT

#include <iostream>
#include <fstream>

// Include this header file to get access to VectorNav sensors
#include "vn/sensors.h"
// We need this file for our sleep function.
#include "vn/thread.h"
// Header for RTIMULib, MPU9250
#include "RTIMULib.h"
// Camera Pose Estimation
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
// Multithreading
#include <vector>
#include <thread>
// For observer calculation
#include <algorithm>
#include <iterator>
#include <string>
#include <math.h>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint.hpp>
#include <Eigen/Dense>
#include <boost/assign/std/vector.hpp> 
#include <boost/random.hpp>
#include <boost/assert.hpp>
#include <boost/algorithm/string.hpp>
#include <unsupported/Eigen/MatrixFunctions>


using namespace std;
using namespace vn::math;
using namespace vn::sensors;
using namespace vn::protocol::uart;
using namespace vn::xplat;

using namespace Eigen;
using namespace boost::numeric::odeint;
typedef std::vector<double, Eigen::aligned_allocator<double>> vec_d;



// Method declaration for future use.
void asciiAsyncMessageReceived(void* userData, Packet& p, size_t index);
void asciiOrBinaryAsyncMessageReceived(void* userData, Packet& p, size_t index);
void updateCamera();
void updateMPU9250();
void updateVN100();
void updateMahonyObserver(double t, double dt);
void updateProposedObserver(double t, double dt);
double getBiasError(string obs_type);
double getAttitudeError(string obs_type);
Matrix3d getRotationMatrix(double yaw, double pitch, double roll);

// Camera Variables
cv::VideoCapture in_video;
cv::Mat image, image_copy;
float actual_marker_length;
cv::Ptr<cv::aruco::Dictionary> dictionary;
cv::Mat camera_matrix, dist_coeffs;
std::ostringstream vector_to_marker;

// MPU9205 Variables
int sampleCount = 0;
int sampleRate = 0;
uint64_t rateTimer=0;
uint64_t displayTimer=0;
uint64_t now;
RTIMU *imu;

// VN100 Variables
VnSensor vs;

// variable to write in csv
double m_time;
float omega_x, omega_y, omega_z;
float cam1_x, cam1_y, cam1_z;
float cam2_x, cam2_y, cam2_z;
float g_x, g_y, g_z;
float true_omega_x, true_omega_y, true_omega_z;
float true_attitude_x, true_attitude_y, true_attitude_z;

//Observer variables
static double ki = 20;
static double kp = 4;
static Matrix3d W;
static Matrix<double,3,1> e1;
static Matrix<double,3,1> e2;
static Matrix<double,3,1> e3;
static Matrix3d S;
static Matrix3d F;
static vec_d Y_init_proposed;
static vec_d Y_init_mahony;
static runge_kutta_dopri5<vec_d> stepper1;
static runge_kutta_dopri5<vec_d> stepper2;
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

Matrix<double, 3, 1> Omega(){
    MatrixXd Omega_t(3,1);
    Omega_t(0,0) = omega_x;
    Omega_t(1,0) = omega_y;
    Omega_t(2,0) = omega_z;

    return Omega_t;
}

Matrix3d R(){
    Matrix3d R_t;
    R_t << cam1_x, cam1_y, cam1_z, cam2_x, cam2_y, cam2_z, g_x, g_y, g_z;
    return R_t;
}

Matrix3d C(){
    Matrix3d R_t = R();
    Matrix3d C_t = R_t.transpose() * S;
    return C_t;
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

    //C_t = C(t);
    Matrix3d C_t = C();

    //A_t = S * W * C_t';
    Matrix3d A_t = S * W * (C_t.transpose());

    //assing dAbardt
    dAbardt = Abar * hat_map(Omega()) - A_t * hat_map(bbar) + kp * (A_t - Abar);

    //for n = 1:3
    //    dbbardt = dbbardt - ki * W(n,n) * cross(C_t(:,n), Abar' * S(:,n)); 
    //end
    dbbardt << 0.0, 0.0, 0.0;
    for (int i = 0 ; i < 3 ; i++){
        //(C_t.col(i)).cross(Abar.transpose() * (S.col(i)));
        Vector3d vec1;
        vec1 = C_t.col(i);
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
    Matrix3d R_t = R();

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
    dRbardt = Rbar * (hat_map(Omega() - bbar) + kp * hat_map(wmes));
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

int main(int argc, char *argv[])
{
	
	/******************
	 * Camera Settings*
	 ******************/
	std::cout << "------------INIT CAMERA...-------------" << std::endl;
	int wait_time = 10;
    actual_marker_length = 0.40;  // this should be in meters
	//cv::Mat image, image_copy;
    //cv::Mat camera_matrix, dist_coeffs;
    //std::ostringstream vector_to_marker;
    
    //cv::VideoCapture in_video;
    in_video.open(1);
    dictionary = 
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
    
    cv::FileStorage fs("/home/nvidia/aruco-markers/calibration_params.yml", cv::FileStorage::READ);

    fs["camera_matrix"] >> camera_matrix;
    fs["distortion_coefficients"] >> dist_coeffs;//#C73C3C#E9C8C8ffs;

    std::cout << "camera_matrix\n" << camera_matrix << std::endl;
    std::cout << "\ndist coeffs\n" << dist_coeffs << std::endl;
    in_video.grab();

	/*******************
	 * MPU9250 Settings*
	 *******************/
	std::cout << "------------INIT MPU9250...-------------" << std::endl;

	RTIMUSettings *settings = new RTIMUSettings("/home/nvidia/observer_experiment","RTIMULib");
	imu = RTIMU::createIMU(settings);
	if((imu==NULL) || (imu->IMUType() == RTIMU_TYPE_NULL)){
		cout << "No IMU Found\n";
		exit(1);
	}

	imu->IMUInit();
	//imu->setSlerpPower(0.02);
	imu->setGyroEnable(true);
	imu->setAccelEnable(true);
	//imu->setCompassEnable(true);
	cout << imu->IMUName() << endl;
	cout << imu->IMUGetPollInterval() << endl;
	/*******************
	 * VN-100 Settings *
	 *******************/
	std::cout << "------------INIT VN100...-------------" << std::endl;
	const string SensorPort = "/dev/ttyUSB0"; // Linux Format
	const uint32_t SensorBaudrate = 115200;
	// create VnSensor Object
	//VnSensor vs;
	vs.connect(SensorPort, SensorBaudrate);

	// query the sensor's model number
	string mn = vs.readModelNumber();
	cout << "Model Number : " << mn << endl;
	
	// Let's do some simple reconfiguration fo the sensor. As it comes from the
	// factory, the sensor outputs asynchronous data at 40Hz. We will change
	// this to 50 Hz for demonstration purposes.
	vs.writeAsyncDataOutputFrequency(50);
	uint32_t newHz = vs.readAsyncDataOutputFrequency();
	cout << "new async frequency: " << newHz << " Hz" << endl;

	// change the heading mode used by the sensor.
	VpeBasicControlRegister vpeReg = vs.readVpeBasicControl();
	vpeReg.headingMode = HEADINGMODE_ABSOLUTE;
	vs.writeVpeBasicControl(vpeReg);
	vpeReg = vs.readVpeBasicControl();
	cout << "new Heading Mode: " << vpeReg.headingMode << endl;

	/************************
	 * 
	 * Observer variable init
	 * 
	 ************************/
	 
	W << 1.0/3, 0.0, 0.0, 0.0, 1.0/3, 0.0, 0.0, 0.0, 1.0/3;
    e1 << 1.0, 0.0, 0.0;
    e2 << 0.0, 1.0, 0.0;
    e3 << 0.0, 0.0, 1.0;

    Matrix<double,3,1> s1;
    Matrix<double,3,1> s2;
    Matrix<double,3,1> s3;
    //s1 = e1;
    //s2 = e2;
    //s3 = e3;
    
    s1 = 0.000*e1+0.149*e2+0.989*e3;
    s2 = 0.162*e1+0.056*e2+0.985*e3;
    s3 = 0.470*e1+0.882*e2+0.020*e3;
    s1 = s1/s1.norm();
    s2 = s2/s2.norm();
    s3 = s3/s3.norm();
    S << s1,s2,s3;
    std::cout << "S is " << S << std::endl;
    std::cout << "W is " << W << std::endl;    
    F = S * W * S.transpose();
    std::cout << "F is " << F << std::endl;

    vec_d bbar_init {0.000000, 0.000000, 0.000000 };
    MatrixXd mat_Rbar_init(3,3);
    mat_Rbar_init << 0.2440,0.9107,-0.3333,
                    0.9107,-0.3333,-0.2440,
                    -0.3333,-0.2440,-0.9107;


    MatrixXd mat_Abar_init(3,3);
    mat_Abar_init = F * mat_Rbar_init;
    vec_d Abar_init(mat_Abar_init.data(), mat_Abar_init.data() + mat_Abar_init.size());
    Y_init_proposed.reserve(12);
    Y_init_proposed.insert( Y_init_proposed.end(), Abar_init.begin(), Abar_init.end() );
    Y_init_proposed.insert( Y_init_proposed.end(), bbar_init.begin(), bbar_init.end() ); 
    Y_init_mahony.reserve(12);
    Y_init_mahony.insert( Y_init_mahony.end(), Abar_init.begin(), Abar_init.end() );
    Y_init_mahony.insert( Y_init_mahony.end(), bbar_init.begin(), bbar_init.end() ); 
	
	
	//std::cout << "Y_init_Proposed is " << Y_init_proposed << std::endl;
	//std::cout << "Y_init_Mahony is " << Y_init_mahony << std::endl;
	std::cout << "Y_mahony is \n";
	for(auto i=Y_init_mahony.begin(); i!=Y_init_mahony.end();++i){
		std::cout << *i << " ";
	}
	std::cout << "\nY_proposed is \n";
	for(auto i=Y_init_proposed.begin(); i!=Y_init_proposed.end();++i){
		std::cout << *i << " ";
	}
	std::cout << "\n";
	
	usleep(5000000);
	
	/*********************
	 * 
	 * Main Loop
	 * 
	 *********************/
	std::ofstream myfile;
	myfile.open("example.csv");
	myfile << "test for writing sensor data\n";
	myfile << "Time, omega_x, omega_y, omega_z,";
	myfile << "cam1_x, cam1_y, cam1_z, cam2_x, cam2_y, cam2_z,";
	myfile << "g_x, g_y, g_z,";
	myfile << "true_omega_x, true_omega_y, true_omega_z,";
	myfile << "true_attitude_x, true_attitude_y, true_attitude_z,";
	myfile << "bias_err_proposed, attitude_err_proposed,";
	myfile << "bias_err_mahony, attitude_err_mahony\n";

	std::cout << "Starting Main Loop... " << std::endl; 
	
	double t, dt;
	std::chrono::time_point<std::chrono::system_clock> t_init, t_prev, t_now;
	t_init = std::chrono::system_clock::now();
	t_prev = t_init;

	while(1){
		// [data acquisition                
		t_now = std::chrono::system_clock::now();
		std::vector<std::thread> ths;
		ths.push_back(thread(&updateCamera));
		ths.push_back(thread(&updateMPU9250));
		ths.push_back(thread(&updateVN100));		
		for(auto& th : ths){
			th.join();
		}
		//]

		// [observer update
		t = (std::chrono::duration<double>(t_now-t_init)).count();
		dt = (std::chrono::duration<double>(t_now-t_prev)).count();
		cout << "t" << t << ", dt" << dt << endl;		
		std::vector<std::thread> observer_ths;
		observer_ths.push_back(thread(&updateProposedObserver, t, dt));
		observer_ths.push_back(thread(&updateMahonyObserver, t, dt));
		for(auto& th : observer_ths){
			th.join();
		}

		//]
		
		// [data logging
		// write in csv file
		m_time = t;		
		myfile << m_time << ",";
		myfile << omega_x << "," << omega_y << "," << omega_z << ",";
		myfile << cam1_x << "," << cam1_y << "," << cam1_z << ",";
		myfile << cam2_x << "," << cam2_y << "," << cam2_z << ",";
		myfile << g_x << "," << g_y << "," << g_z << ",";
		myfile << true_omega_x << "," << true_omega_y << "," << true_omega_z << ",";
		myfile << true_attitude_x << "," << true_attitude_y << "," << true_attitude_z << ",";

		// write bias error in csv file
		double bias_err_proposed = getBiasError("proposed");
		myfile << bias_err_proposed << ",";
		// write atttiude error in csv file
		double attitude_err_proposed = getAttitudeError("proposed");
		myfile << attitude_err_proposed << ",";

		// write bias error in csv file
		double bias_err_mahony = getBiasError("mahony");
		myfile << bias_err_mahony << ",";
		// write atttiude error in csv file
		double attitude_err_mahony = getAttitudeError("mahony");
		myfile << attitude_err_mahony << "\n";

		// [Loop exit condition
		std::cout << "one loop takes : " << t << " s\n";
		if(t > 10){
			std::cout << "10 sec passed and quit\n";
			break;
		}
		t_prev = t_now;
		//] 
 
	}
	//myfile.close();
    in_video.release();
	////////////////////////////////////////////////////////////////

	vs.disconnect();
	return 0;
}

double getBiasError(string obs_type){
	vec_d Y_init;
	if(obs_type.compare("mahony")==0){
		Y_init = Y_init_mahony;
	}
	else{
		Y_init = Y_init_proposed;
	}
	double bias_bar_x = Y_init[9];
	double bias_bar_y = Y_init[10];
	double bias_bar_z = Y_init[11];
	std::cout << bias_bar_x << bias_bar_y << bias_bar_y << std::endl;
	double bias_x = -true_omega_x + omega_x;
	double bias_y = -true_omega_y + omega_y;
	double bias_z = -true_omega_z + omega_z;
	double retval;
	retval += (bias_x-bias_bar_x)*(bias_x-bias_bar_x);
	retval += (bias_y-bias_bar_y)*(bias_y-bias_bar_y);
	retval += (bias_z-bias_bar_z)*(bias_z-bias_bar_z);
	retval = sqrt(retval);
	return retval;
}

double getAttitudeError(string obs_type){
	vec_d Y_init;
	if(obs_type.compare("mahony")==0){
		Y_init = Y_init_mahony;
	}
	else{
		Y_init = Y_init_proposed;
	}
	double retval;
	Matrix3d true_attitude = getRotationMatrix(true_attitude_x, true_attitude_y, true_attitude_z);
	Matrix3d estimate_attitude;
	estimate_attitude << Y_init[0], Y_init[1], Y_init[2],
						 Y_init[3], Y_init[4], Y_init[5],
						 Y_init[6], Y_init[7], Y_init[8];
	estimate_attitude = F.inverse() * estimate_attitude;
	retval = (true_attitude - estimate_attitude).norm();
	return retval;
}

Matrix3d getRotationMatrix(double yaw, double pitch, double roll){
	Matrix3d retMatrix;
	retMatrix(0,0) = cos(yaw)*cos(pitch);
	retMatrix(1,0) = sin(yaw)*cos(pitch);
	retMatrix(2,0) = -sin(pitch);
	retMatrix(0,1) = cos(yaw)*sin(pitch)*sin(roll)-sin(yaw)*cos(roll);
	retMatrix(1,1) = sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll);
	retMatrix(2,1) = cos(pitch)*sin(roll);
	retMatrix(0,2) = cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll);
	retMatrix(1,2) = sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll);
	retMatrix(2,2) = cos(pitch)*cos(roll);
	return retMatrix;
}


void asciiAsyncMessageReceived(void* userData, Packet& p, size_t index)
{
	if (p.type() != Packet::TYPE_ASCII)
		return;
	
	if (p.determineAsciiAsyncType() != VNQMR)
		return;

	vec4f quat;
	vec3f mag, accel, ar;

	p.parseVNQMR(&quat, &mag, &accel, &ar);

	cout << "ASCII Async QUAT: " << quat << endl;
	cout << "ASCII Async MAG: " << mag << endl;
	cout << "ASCII Async ACCEL: " << accel << endl;
	cout << "ASCII Async AR: " << ar << endl<<endl;
}

void updateProposedObserver(double t, double dt){
	stepper1.do_step( proposed_observer, Y_init_proposed, t, dt );
}

void updateMahonyObserver(double t, double dt){
	stepper2.do_step( mahony_observer, Y_init_mahony, t, dt );
}

void updateCamera(){
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    if (in_video.grab()) 
    {
        in_video.retrieve(image);
        image.copyTo(image_copy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);        
        // if at least one marker detected
        if (ids.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(image_copy, corners, ids);
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, actual_marker_length,
                    camera_matrix, dist_coeffs, rvecs, tvecs);
            // draw axis for each marker
            for(int i=0; i < ids.size(); i++)
            {                
                if(ids[i]==100){
				    cam1_x = tvecs[i](0);
				    cam1_y = tvecs[i](1);
				    cam1_z = tvecs[i](2);
				    float magCam1 = sqrt(cam1_x*cam1_x+cam1_y*cam1_y+cam1_z*cam1_z);
				    cam1_x /= magCam1;
				    cam1_y /= magCam1;
				    cam1_z /= magCam1;
				}
				else if (ids[i]==101){
					cam2_x = tvecs[i](0);
					cam2_y = tvecs[i](1);
					cam2_z = tvecs[i](2);
				    float magCam2 = sqrt(cam2_x*cam2_x+cam2_y*cam2_y+cam2_z*cam2_z);
				    cam2_x /= magCam2;
				    cam2_y /= magCam2;
				    cam2_z /= magCam2;					
				}
				else{
				    //do nothing
				}
                // print pose estimation
                std::cout << "-------------" << ids[i] << " marker " <<  "-----------\n";
                std::cout << "tvecs\t" << tvecs[i](0) << "\t" << tvecs[i](1) << "\t" << tvecs[i](2) << "\n\n\n\n";  
			}
        }        
    }
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end-start;
	std::cout << "cam thread takes : " << elapsed_time.count() << " s\n";
}

void updateMPU9250(){
	
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	while(!imu->IMURead()){
		printf("*\n");
	}
	RTIMU_DATA imuData = imu->getIMUData();
	printf("MPU Value : %s\n", RTMath::displayDegrees("",imuData.gyro));
	printf("MPU time  : %d\n", (int)imuData.timestamp);
	fflush(stdout);
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end-start;
	std::cout << "MPU9250 thread takes : " << elapsed_time.count() << " s\n";
	
	omega_x = imuData.gyro.data(0);
	omega_y = imuData.gyro.data(1);
	omega_z = imuData.gyro.data(2);
	if(omega_x > 1){
		omega_x = 1;
	}
	else if(omega_x < -1){
		omega_x = -1;
	}

	if(omega_y > 1){
		omega_y = 1;
	}
	else if(omega_y < -1){
		omega_y = -1;
	}
	
	if(omega_z > 1){
		omega_z = 1;
	}
	else if(omega_z < -1){
		omega_z = -1;
	}
	
	////Wait until IMU can get data
	//while(1){
		//int count=0;
		//printf("*\n");

		//while(imu->IMURead()){
			//count++;
			////if(imu->IMURead()){
			//RTIMU_DATA imuData = imu->getIMUData();
			//printf("MPU Value : %s\n", RTMath::displayDegrees("",imuData.gyro));
			//printf("MPU time  : %d\n", (int)imuData.timestamp);
			//fflush(stdout);
			//printf("count : %d\n",count);
			//std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
			//std::chrono::duration<double> elapsed_time = end-start;
			//std::cout << "MPU9250 thread takes : " << elapsed_time.count() << " s\n";
		//}

		
	//}
		//sampleCount++;
		//now = RTMath::currentUSecsSinceEpoch();
		//if((now-displayTimer)>100000){
			//printf("Sample rate %d: %s\n", sampleRate,  RTMath::displayDegrees("",imuData.gyro));
			//fflush(stdout);
			//displayTimer = now;
		//}

		//if((now- rateTimer) > 1000000) {
			//sampleRate = sampleCount;
			//sampleCount = 0;
			//rateTimer = now;
		//}
	//}

}

void updateVN100(){
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

	YawPitchRollMagneticAccelerationAndAngularRatesRegister reg;
	reg = vs.readYawPitchRollMagneticAccelerationAndAngularRates();
	cout << "VN current YPR: " << reg.yawPitchRoll << endl;
	cout << "VN current Acc: " << reg.accel << endl;	
	cout << "VN current Omg: " << reg.gyro << endl;

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end-start;
	std::cout << "vn100 thread takes : " << elapsed_time.count() << " s\n";
	
	true_attitude_x = reg.yawPitchRoll.c[0];
	true_attitude_y = reg.yawPitchRoll.c[1];
	true_attitude_z = reg.yawPitchRoll.c[2];
	true_omega_x = reg.gyro.c[0];
	true_omega_y = reg.gyro.c[1];
	true_omega_z = reg.gyro.c[2];
	//Artificial Bias
	//omega_x = true_omega_x + 1;
	//omega_y = true_omega_x - 1;
	//omega_z = true_omega_x - 1;		
	g_x = -reg.accel.c[0];
	g_y = -reg.accel.c[1];
	g_z = -reg.accel.c[2];
	float magG = sqrt(g_x*g_x+g_y*g_y+g_z*g_z);
	g_x /= magG;
	g_y /= magG;	
	g_z /= magG;
}


void asciiOrBinaryAsyncMessageReceived(void* userData, Packet& p, size_t index)
{
	if (p.type() == Packet::TYPE_ASCII && p.determineAsciiAsyncType() == VNYPR)
	{
		vec3f ypr;
		p.parseVNYPR(&ypr);
		cout << "ASCII Async YPR: " << ypr << endl;
		return;
	}

	if (p.type() == Packet::TYPE_BINARY)
	{
		if(!p.isCompatible(
					COMMONGROUP_TIMESTARTUP | COMMONGROUP_YAWPITCHROLL,
					TIMEGROUP_NONE,
					IMUGROUP_NONE,
					GPSGROUP_NONE,
					ATTITUDEGROUP_NONE,
					INSGROUP_NONE))
			return;

		uint64_t timeStartup = p.extractUint64();
		vec3f ypr = p.extractVec3f();
		cout << "Binary Async TimeStartup: " << timeStartup << endl;
		cout << "Binary Async YPR: " << ypr << endl;
	}
}
