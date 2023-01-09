// ---------------------------------------------------------------------
// RNN4AP project
// Copyright (C) 2021-2022 ISAE
// 
// Purpose:
// Evaluation of Recurrent Neural Networks for future Autopilot Systems
//
// Contact:
// jean-baptiste.chaudron@isae-supaero.fr
// goncalo.fontes-neves@student.isae-supaero.fr
// ---------------------------------------------------------------------

#include <px4_log_handler.hh>

//---------------------------------------------------------------------
px4_log_handler::px4_log_handler()
{
		_mag_x_adds.push_back(1000.000); // min
		_mag_x_adds.push_back(-1000.000); // max
		_mag_x_adds.push_back(0.0); // avg
		_mag_x_adds.push_back(0.0); // std
		_mag_x_adds.push_back(0.0); // prev avg
		
		_mag_y_adds.push_back(1000.000); // min
		_mag_y_adds.push_back(-1000.000); // max
		_mag_y_adds.push_back(0.0); // avg
		_mag_y_adds.push_back(0.0); // std
		_mag_y_adds.push_back(0.0); // prev avg
		
		_mag_z_adds.push_back(1000.000); // min
		_mag_z_adds.push_back(-1000.000); // max
		_mag_z_adds.push_back(0.0); // avg
		_mag_z_adds.push_back(0.0); // std
		_mag_z_adds.push_back(0.0); // prev avg
		
		_acc_x_adds.push_back(1000.000); // min
		_acc_x_adds.push_back(-1000.000); // max
		_acc_x_adds.push_back(0.0); // avg
		_acc_x_adds.push_back(0.0); // std
		_acc_x_adds.push_back(0.0); // prev avg
		
		_acc_y_adds.push_back(1000.000); // min
		_acc_y_adds.push_back(-1000.000); // max
		_acc_y_adds.push_back(0.0); // avg
		_acc_y_adds.push_back(0.0); // std
		_acc_y_adds.push_back(0.0); // prev avg
		
		_acc_z_adds.push_back(1000.000); // min
		_acc_z_adds.push_back(-1000.000); // max
		_acc_z_adds.push_back(0.0); // avg
		_acc_z_adds.push_back(0.0); // std
		_acc_z_adds.push_back(0.0); // prev avg
		
		_gyr_x_adds.push_back(1000.000); // min
		_gyr_x_adds.push_back(-1000.000); // max
		_gyr_x_adds.push_back(0.0); // avg
		_gyr_x_adds.push_back(0.0); // std
		_gyr_x_adds.push_back(0.0); // prev avg
		
		_gyr_y_adds.push_back(1000.000); // min
		_gyr_y_adds.push_back(-1000.000); // max
		_gyr_y_adds.push_back(0.0); // avg
		_gyr_y_adds.push_back(0.0); // std
		_gyr_y_adds.push_back(0.0); // prev avg
		
		_gyr_z_adds.push_back(1000.000); // min
		_gyr_z_adds.push_back(-1000.000); // max
		_gyr_z_adds.push_back(0.0); // avg
		_gyr_z_adds.push_back(0.0); // std
		_gyr_z_adds.push_back(0.0); // prev avg
		
		_pitch_adds.push_back(1000.000); // min
		_pitch_adds.push_back(-1000.000); // max
		_pitch_adds.push_back(0.0); // avg
		_pitch_adds.push_back(0.0); // std
		_pitch_adds.push_back(0.0); // prev avg
		
		_roll_adds.push_back(1000.000); // min
		_roll_adds.push_back(-1000.000); // max
		_roll_adds.push_back(0.0); // avg
		_roll_adds.push_back(0.0); // std
		_roll_adds.push_back(0.0); // prev avg
		
		_yaw_adds.push_back(1000.000); // min
		_yaw_adds.push_back(-1000.000); // max
		_yaw_adds.push_back(0.0); // avg
		_yaw_adds.push_back(0.0); // std
		_yaw_adds.push_back(0.0); // prev avg
		
		
		_mag_x_adds_stdscore.push_back(1000.000); // min
		_mag_x_adds_stdscore.push_back(-1000.000); // max
		_mag_x_adds_stdscore.push_back(0.0); // avg
		_mag_x_adds_stdscore.push_back(0.0); // std
		_mag_x_adds_stdscore.push_back(0.0); // prev avg
		
		_mag_y_adds_stdscore.push_back(1000.000); // min
		_mag_y_adds_stdscore.push_back(-1000.000); // max
		_mag_y_adds_stdscore.push_back(0.0); // avg
		_mag_y_adds_stdscore.push_back(0.0); // std
		_mag_y_adds_stdscore.push_back(0.0); // prev avg
		
		_mag_z_adds_stdscore.push_back(1000.000); // min
		_mag_z_adds_stdscore.push_back(-1000.000); // max
		_mag_z_adds_stdscore.push_back(0.0); // avg
		_mag_z_adds_stdscore.push_back(0.0); // std
		_mag_z_adds_stdscore.push_back(0.0); // prev avg
		
		_acc_x_adds_stdscore.push_back(1000.000); // min
		_acc_x_adds_stdscore.push_back(-1000.000); // max
		_acc_x_adds_stdscore.push_back(0.0); // avg
		_acc_x_adds_stdscore.push_back(0.0); // std
		_acc_x_adds_stdscore.push_back(0.0); // prev avg
		
		_acc_y_adds_stdscore.push_back(1000.000); // min
		_acc_y_adds_stdscore.push_back(-1000.000); // max
		_acc_y_adds_stdscore.push_back(0.0); // avg
		_acc_y_adds_stdscore.push_back(0.0); // std
		_acc_y_adds_stdscore.push_back(0.0); // prev avg
		
		_acc_z_adds_stdscore.push_back(1000.000); // min
		_acc_z_adds_stdscore.push_back(-1000.000); // max
		_acc_z_adds_stdscore.push_back(0.0); // avg
		_acc_z_adds_stdscore.push_back(0.0); // std
		_acc_z_adds_stdscore.push_back(0.0); // prev avg
		
		_gyr_x_adds_stdscore.push_back(1000.000); // min
		_gyr_x_adds_stdscore.push_back(-1000.000); // max
		_gyr_x_adds_stdscore.push_back(0.0); // avg
		_gyr_x_adds_stdscore.push_back(0.0); // std
		_gyr_x_adds_stdscore.push_back(0.0); // prev avg
		
		_gyr_y_adds_stdscore.push_back(1000.000); // min
		_gyr_y_adds_stdscore.push_back(-1000.000); // max
		_gyr_y_adds_stdscore.push_back(0.0); // avg
		_gyr_y_adds_stdscore.push_back(0.0); // std
		_gyr_y_adds_stdscore.push_back(0.0); // prev avg
		
		_gyr_z_adds_stdscore.push_back(1000.000); // min
		_gyr_z_adds_stdscore.push_back(-1000.000); // max
		_gyr_z_adds_stdscore.push_back(0.0); // avg
		_gyr_z_adds_stdscore.push_back(0.0); // std
		_gyr_z_adds_stdscore.push_back(0.0); // prev avg
		
		_pitch_adds_stdscore.push_back(1000.000); // min
		_pitch_adds_stdscore.push_back(-1000.000); // max
		_pitch_adds_stdscore.push_back(0.0); // avg
		_pitch_adds_stdscore.push_back(0.0); // std
		_pitch_adds_stdscore.push_back(0.0); // prev avg
		
		_roll_adds_stdscore.push_back(1000.000); // min
		_roll_adds_stdscore.push_back(-1000.000); // max
		_roll_adds_stdscore.push_back(0.0); // avg
		_roll_adds_stdscore.push_back(0.0); // std
		_roll_adds_stdscore.push_back(0.0); // prev avg
		
		_yaw_adds_stdscore.push_back(1000.000); // min
		_yaw_adds_stdscore.push_back(-1000.000); // max
		_yaw_adds_stdscore.push_back(0.0); // avg
		_yaw_adds_stdscore.push_back(0.0); // std
		_yaw_adds_stdscore.push_back(0.0); // prev avg
		
		_nb_elem = 0;
		
		// for randomizing
		std::srand ( unsigned ( std::time(0) ) );
}

//---------------------------------------------------------------------
px4_log_handler::~px4_log_handler()
{
	
}


//---------------------------------------------------------------------
void px4_log_handler::load_log(const char *px4logfilename)
{
	// stream for the log
    std::ifstream px4logfile; 
    // string to store each file line
    std::string line; 
    char delim;
    // tmp data
    float timestamp_ts = 0.0, period_us = 0.0, exec_us= 0.0;
    float x= 0.0, y= 0.0, z= 0.0;
    float vx= 0.0, vy= 0.0, vz= 0.0;
    float roll_ot= 0.0, pitch_ot= 0.0, yaw_ot= 0.0;
    float roll_px4= 0.0, pitch_px4= 0.0, yaw_px4= 0.0;
    float acc_x= 0.0, acc_y= 0.0, acc_z= 0.0;
    float gyr_x= 0.0, gyr_y= 0.0, gyr_z= 0.0;
    float mag_x= 0.0, mag_y= 0.0, mag_z= 0.0;
    float abs_pres= 0.0, ult_dist= 0.0;
        
    px4logfile.open(px4logfilename);
    
    if (!px4logfile.is_open()) 
    {
        std::cerr << "There was a problem opening the PX4 LOG input file " << px4logfilename << std::endl;
        exit(1);
    }
    else
    {
		std::getline(px4logfile, line); // read the first line only for column title
	}
    while(std::getline(px4logfile, line))
	{
		_it_vector.push_back(_nb_elem);
		//_random_it_vector.push_back(_nb_elem);
		_nb_elem++;
		std::stringstream lineStream(line);
		 lineStream >> timestamp_ts >> delim >> period_us >> delim >> exec_us >> delim  
		>> x >> delim >> y >> delim  >> z >> delim 
		>> vx >> delim >> vy >> delim >> vz >> delim
		>> roll_ot >> delim >> pitch_ot >> delim >> yaw_ot >> delim 
		>> roll_px4 >> delim >> pitch_px4 >> delim >> yaw_px4 >> delim 
		>> acc_x >> delim >> acc_y >> delim >> acc_z >> delim 
		>> gyr_x >> delim >> gyr_y >> delim >> gyr_z >> delim
		>> mag_x >> delim >> mag_y >> delim >> mag_z >> delim
		>> abs_pres >> delim >> ult_dist;
		
		// put data in corresponding vector
		// inputs
		_acc_x.push_back(acc_x);
		_acc_y.push_back(acc_y);
		_acc_z.push_back(acc_z);
		_gyr_x.push_back(gyr_x);
		_gyr_y.push_back(gyr_y);
		_gyr_z.push_back(gyr_z);
		_mag_x.push_back(mag_x);
		_mag_y.push_back(mag_y);
		_mag_z.push_back(mag_z);
		// labels
		_pitch.push_back(pitch_px4);
		_roll.push_back(roll_px4);
		_yaw.push_back(yaw_px4);
	}
	px4logfile.close();
	//std::cerr << "_nb_elem=" << _nb_elem << std::endl;
}

//---------------------------------------------------------------------
void px4_log_handler::load_log11(const char *px4logfilename)
{
	// stream for the log
    std::ifstream px4logfile; 
    // string to store each file line
    std::string line; 
    char delim;
    // tmp data
    float timestamp_ts = 0.0, period_us = 0.0, exec_us= 0.0;
    float x= 0.0, y= 0.0, z= 0.0;
    float vx= 0.0, vy= 0.0, vz= 0.0;
    float roll_ot= 0.0, pitch_ot= 0.0, yaw_ot= 0.0;
    float roll_px4= 0.0, pitch_px4= 0.0, yaw_px4= 0.0;
    float acc_x= 0.0, acc_y= 0.0, acc_z= 0.0;
    float gyr_x= 0.0, gyr_y= 0.0, gyr_z= 0.0;
    float mag_x= 0.0, mag_y= 0.0, mag_z= 0.0;
    float abs_pres= 0.0, ult_dist= 0.0;
        
    px4logfile.open(px4logfilename);
    
    if (!px4logfile.is_open()) 
    {
        std::cerr << "There was a problem opening the PX4 LOG input file!\n" << std::endl;
        exit(1);
    }
    else
    {
		std::getline(px4logfile, line); // read the first line only for column title
	}
    while(std::getline(px4logfile, line))
	{
		_it_vector.push_back(_nb_elem);
		//_random_it_vector.push_back(_nb_elem);
		_nb_elem++;
		std::stringstream lineStream(line);
		 lineStream >> timestamp_ts >> delim 
		>> roll_ot >> delim >> pitch_ot >> delim >> yaw_ot >> delim 
		>> roll_px4 >> delim >> pitch_px4 >> delim >> yaw_px4 >> delim 
		>> acc_x >> delim >> acc_y >> delim >> acc_z >> delim 
		>> gyr_x >> delim >> gyr_y >> delim >> gyr_z >> delim
		>> mag_x >> delim >> mag_y >> delim >> mag_z >> delim;

		// put data in corresponding vector
		// inputs
		_acc_x.push_back(acc_x);
		_acc_y.push_back(acc_y);
		_acc_z.push_back(acc_z);
		_gyr_x.push_back(gyr_x);
		_gyr_y.push_back(gyr_y);
		_gyr_z.push_back(gyr_z);
		_mag_x.push_back(mag_x);
		_mag_y.push_back(mag_y);
		_mag_z.push_back(mag_z);
		// labels
		_pitch.push_back(pitch_px4);
		_roll.push_back(roll_px4);
		_yaw.push_back(yaw_px4);
	}
	px4logfile.close();
	//std::cerr << "_nb_elem=" << _nb_elem << std::endl;
}


//---------------------------------------------------------------------
void px4_log_handler::update_adds()
{
	uint32_t i1;
	float tmp = 0.0;
	// loop from 1 to avoid division per 0, not really correct but it does the job
	for (i1 = 1; i1 < _nb_elem; ++i1)
	{
		if(_mag_x[i1] < _mag_x_adds[0])
			_mag_x_adds[0] = _mag_x[i1]; // min
		if(_mag_x[i1] > _mag_x_adds[1])
			_mag_x_adds[1] = _mag_x[i1]; // max
		_mag_x_adds[4] = _mag_x_adds[2];
		_mag_x_adds[2] = _mag_x_adds[2] + (_mag_x[i1]-_mag_x_adds[2])/i1; //avg
		_mag_x_adds[3] = std::sqrt(((i1 - 1.0) * _mag_x_adds[3] * _mag_x_adds[3] + (_mag_x[i1] - _mag_x_adds[4]) * (_mag_x[i1] - _mag_x_adds[2]))/i1);
		
		if(_mag_y[i1] < _mag_y_adds[0])
			_mag_y_adds[0] = _mag_y[i1]; // min
		if(_mag_y[i1] > _mag_y_adds[1])
			_mag_y_adds[1] = _mag_y[i1]; // max
		
		_mag_y_adds[4] = _mag_y_adds[2];
		_mag_y_adds[2] = _mag_y_adds[2] + (_mag_y[i1]-_mag_y_adds[2])/i1; //avg
		_mag_y_adds[3] = std::sqrt(((i1 - 1.0) * _mag_y_adds[3] * _mag_y_adds[3] + (_mag_y[i1] - _mag_y_adds[4]) * (_mag_y[i1] - _mag_y_adds[2]))/i1);

			
		if(_mag_z[i1] < _mag_z_adds[0])
			_mag_z_adds[0] = _mag_z[i1]; // min
		if(_mag_z[i1] > _mag_z_adds[1])
			_mag_z_adds[1] = _mag_z[i1]; // max
		_mag_z_adds[4] = _mag_z_adds[2];
		_mag_z_adds[2] = _mag_z_adds[2] + (_mag_z[i1]-_mag_z_adds[2])/i1; //avg
		_mag_z_adds[3] = std::sqrt(((i1 - 1.0) * _mag_z_adds[3] * _mag_z_adds[3] + (_mag_z[i1] - _mag_z_adds[4]) * (_mag_z[i1] - _mag_z_adds[2]))/i1);
			
		if(_acc_x[i1] < _acc_x_adds[0])
			_acc_x_adds[0] = _acc_x[i1]; // min
		if(_acc_x[i1] > _acc_x_adds[1])
			_acc_x_adds[1] = _acc_x[i1]; // max
		_acc_x_adds[4] = _acc_x_adds[2];
		_acc_x_adds[2] = _acc_x_adds[2] + (_acc_x[i1]-_acc_x_adds[2])/i1; //avg
		_acc_x_adds[3] = std::sqrt(((i1 - 1.0) * _acc_x_adds[3] * _acc_x_adds[3] + (_acc_x[i1] - _acc_x_adds[4]) * (_acc_x[i1] - _acc_x_adds[2]))/i1);

		if(_acc_y[i1] < _acc_y_adds[0])
			_acc_y_adds[0] = _acc_y[i1]; // min
		if(_acc_y[i1] > _acc_y_adds[1])
			_acc_y_adds[1] = _acc_y[i1]; // max
		_acc_y_adds[4] = _acc_y_adds[2];
		_acc_y_adds[2] = _acc_y_adds[2] + (_acc_y[i1]-_acc_y_adds[2])/i1; //avg
		_acc_y_adds[3] = std::sqrt(((i1 - 1.0) * _acc_y_adds[3] * _acc_y_adds[3] + (_acc_y[i1] - _acc_y_adds[4]) * (_acc_y[i1] - _acc_y_adds[2]))/i1);
			
		if(_acc_z[i1] < _acc_z_adds[0])
			_acc_z_adds[0] = _acc_z[i1]; // min
		if(_acc_z[i1] > _acc_z_adds[1])
			_acc_z_adds[1] = _acc_z[i1]; // max
		_acc_z_adds[4] = _acc_z_adds[2];
		_acc_z_adds[2] = _acc_z_adds[2] + (_acc_z[i1]-_acc_z_adds[2])/i1; //avg
		_acc_z_adds[3] = std::sqrt(((i1 - 1.0) * _acc_z_adds[3] * _acc_z_adds[3] + (_acc_z[i1] - _acc_z_adds[4]) * (_acc_z[i1] - _acc_z_adds[2]))/i1);
		
		if(_gyr_x[i1] < _gyr_x_adds[0])
			_gyr_x_adds[0] = _gyr_x[i1]; // min
		if(_gyr_x[i1] > _gyr_x_adds[1])
			_gyr_x_adds[1] = _gyr_x[i1]; // max
		_gyr_x_adds[4] = _gyr_x_adds[2];
		_gyr_x_adds[2] = _gyr_x_adds[2] + (_gyr_x[i1]-_gyr_x_adds[2])/i1; //avg
		_gyr_x_adds[3] = std::sqrt(((i1 - 1.0) * _gyr_x_adds[3] * _gyr_x_adds[3] + (_gyr_x[i1] - _gyr_x_adds[4]) * (_gyr_x[i1] - _gyr_x_adds[2]))/i1);
			
		if(_gyr_y[i1] < _gyr_y_adds[0])
			_gyr_y_adds[0] = _gyr_y[i1]; // min
		if(_gyr_y[i1] > _gyr_y_adds[1])
			_gyr_y_adds[1] = _gyr_y[i1]; // max
		_gyr_y_adds[4] = _gyr_y_adds[2];
		_gyr_y_adds[2] = _gyr_y_adds[2] + (_gyr_y[i1]-_gyr_y_adds[2])/i1; //avg
		_gyr_y_adds[3] = std::sqrt(((i1 - 1.0) * _gyr_y_adds[3] * _gyr_y_adds[3] + (_gyr_y[i1] - _gyr_y_adds[4]) * (_gyr_y[i1] - _gyr_y_adds[2]))/i1);
			
		if(_gyr_z[i1] < _gyr_z_adds[0])
			_gyr_z_adds[0] = _gyr_z[i1]; // min
		if(_gyr_z[i1] > _gyr_z_adds[1])
			_gyr_z_adds[1] = _gyr_z[i1]; // max
		_gyr_z_adds[4] = _gyr_z_adds[2];
		_gyr_z_adds[2] = _gyr_z_adds[2] + (_gyr_z[i1]-_gyr_z_adds[2])/i1; //avg
		_gyr_z_adds[3] = std::sqrt(((i1 - 1.0) * _gyr_z_adds[3] * _gyr_z_adds[3] + (_gyr_z[i1] - _gyr_z_adds[4]) * (_gyr_z[i1] - _gyr_z_adds[2]))/i1);
			
		if(_pitch[i1] < _pitch_adds[0])
			_pitch_adds[0] = _pitch[i1]; // min
		if(_pitch[i1] > _pitch_adds[1])
			_pitch_adds[1] = _pitch[i1]; // max
		_pitch_adds[4] = _pitch_adds[2];
		_pitch_adds[2] = _pitch_adds[2] + (_pitch[i1]-_pitch_adds[2])/i1; //avg
		_pitch_adds[3] = std::sqrt(((i1 - 1.0) * _pitch_adds[3] * _pitch_adds[3] + (_pitch[i1] - _pitch_adds[4]) * (_pitch[i1] - _pitch_adds[2]))/i1);
			
		if(_roll[i1] < _roll_adds[0])
			_roll_adds[0] = _roll[i1]; // min
		if(_roll[i1] > _roll_adds[1])
			_roll_adds[1] = _roll[i1]; // max
		_roll_adds[4] = _roll_adds[2];
		_roll_adds[2] = _roll_adds[2] + (_roll[i1]-_roll_adds[2])/i1; //avg
		_roll_adds[3] = std::sqrt(((i1 - 1.0) * _roll_adds[3] * _roll_adds[3] + (_roll[i1] - _roll_adds[4]) * (_roll[i1] - _roll_adds[2]))/i1);
			
		if(_yaw[i1] < _yaw_adds[0])
			_yaw_adds[0] = _yaw[i1]; // min
		if(_yaw[i1] > _yaw_adds[1])
			_yaw_adds[1] = _yaw[i1]; // max
		_yaw_adds[4] = _yaw_adds[2];
		_yaw_adds[2] = _yaw_adds[2] + (_yaw[i1]-_yaw_adds[2])/i1; //avg
		_yaw_adds[3] = std::sqrt(((i1 - 1.0) * _yaw_adds[3] * _yaw_adds[3] + (_yaw[i1] - _yaw_adds[4]) * (_yaw[i1] - _yaw_adds[2]))/i1);
	}
}

//---------------------------------------------------------------------
void px4_log_handler::print_adds()
{
	std::cout<< "---------------------------------------" << std::endl;
	std::cout<< "DATA SET WITH CNT=" << _nb_elem << std::endl;
	std::cout<< "---------------------------------------" << std::endl;
	std::cout<< "------ MAG X ----------" << std::endl;
	std::cout<< "min = " << _mag_x_adds[0] << std::endl;
	std::cout<< "max = " << _mag_x_adds[1] << std::endl;
	std::cout<< "mean = " << _mag_x_adds[2] << std::endl;
	std::cout<< "std = " << _mag_x_adds[3] << std::endl;
	std::cout<< "------ MAG Y ----------" << std::endl;
	std::cout<< "min = " << _mag_y_adds[0] << std::endl;
	std::cout<< "max = " << _mag_y_adds[1] << std::endl;
	std::cout<< "mean = " << _mag_y_adds[2] << std::endl;
	std::cout<< "std = " << _mag_y_adds[3] << std::endl;
	std::cout<< "------ MAG Z ----------" << std::endl;
	std::cout<< "min = " << _mag_z_adds[0] << std::endl;
	std::cout<< "max = " << _mag_z_adds[1] << std::endl;
	std::cout<< "mean = " << _mag_z_adds[2] << std::endl;
	std::cout<< "std = " << _mag_z_adds[3] << std::endl;
	std::cout << "" <<  std::endl;
	std::cout<< "------ ACC X ----------" << std::endl;
	std::cout<< "min = " << _acc_x_adds[0] << std::endl;
	std::cout<< "max = " << _acc_x_adds[1] << std::endl;
	std::cout<< "mean = " << _acc_x_adds[2] << std::endl;
	std::cout<< "std = " << _acc_x_adds[3] << std::endl;
	std::cout<< "------ ACC Y ----------" << std::endl;
	std::cout<< "min = " << _acc_y_adds[0] << std::endl;
	std::cout<< "max = " << _acc_y_adds[1] << std::endl;
	std::cout<< "mean = " << _acc_y_adds[2] << std::endl;
	std::cout<< "std = " << _acc_y_adds[3] << std::endl;
	std::cout<< "------ ACC Z ----------" << std::endl;
	std::cout<< "min = " << _acc_z_adds[0] << std::endl;
	std::cout<< "max = " << _acc_z_adds[1] << std::endl;
	std::cout<< "mean = " << _acc_z_adds[2] << std::endl;
	std::cout<< "std = " << _acc_z_adds[3] << std::endl;
	std::cout << "" <<  std::endl;
	std::cout<< "------ PITCH ----------" << std::endl;
	std::cout<< "min = " << _pitch_adds[0] << std::endl;
	std::cout<< "max = " << _pitch_adds[1] << std::endl;
	std::cout<< "mean = " << _pitch_adds[2] << std::endl;
	std::cout<< "std = " << _pitch_adds[3] << std::endl;
	std::cout<< "------ ROLL ----------" << std::endl;
	std::cout<< "min = " << _roll_adds[0] << std::endl;
	std::cout<< "max = " << _roll_adds[1] << std::endl;
	std::cout<< "mean = " << _roll_adds[2] << std::endl;
	std::cout<< "std = " << _roll_adds[3] << std::endl;
	std::cout<< "------ YAW ----------" << std::endl;
	std::cout<< "min = " << _yaw_adds[0] << std::endl;
	std::cout<< "max = " << _yaw_adds[1] << std::endl;
	std::cout<< "mean = " << _yaw_adds[2] << std::endl;
	std::cout<< "std = " << _yaw_adds[3] << std::endl;
}

//---------------------------------------------------------------------
void px4_log_handler::normalize()
{
	uint32_t i1 = 0;
	float tmp = 0.0;
	float sqrt_tmp = 0.0;

	for (i1 = 1; i1 < _nb_elem; i1++)
	{
		_mag_x_stdscore.push_back((_mag_x[i1] - _mag_x_adds[2])/_mag_x_adds[3]);
		_mag_y_stdscore.push_back((_mag_y[i1] - _mag_y_adds[2])/_mag_y_adds[3]);
		_mag_z_stdscore.push_back((_mag_z[i1] - _mag_z_adds[2])/_mag_z_adds[3]);
		
		_acc_x_stdscore.push_back((_acc_x[i1] - _acc_x_adds[2])/_acc_x_adds[3]);
		_acc_y_stdscore.push_back((_acc_y[i1] - _acc_y_adds[2])/_acc_y_adds[3]);
		_acc_z_stdscore.push_back((_acc_z[i1] - _acc_z_adds[2])/_acc_z_adds[3]);
		
		_gyr_x_stdscore.push_back((_gyr_x[i1] - _gyr_x_adds[2])/_gyr_x_adds[3]);
		_gyr_y_stdscore.push_back((_gyr_y[i1] - _gyr_y_adds[2])/_gyr_y_adds[3]);
		_gyr_z_stdscore.push_back((_gyr_z[i1] - _gyr_z_adds[2])/_gyr_z_adds[3]);
		
		_pitch_stdscore.push_back((_pitch[i1] - _pitch_adds[2])/_pitch_adds[3]);
		_roll_stdscore.push_back((_roll[i1] - _roll_adds[2])/_roll_adds[3]);
		_yaw_stdscore.push_back((_yaw[i1] - _yaw_adds[2])/_yaw_adds[3]);
	}
	
	// loop from 1 to avoid division per 0, not really correct but it does the job
	for (i1 = 1; i1 < _nb_elem; ++i1)
	{
		if(_mag_x_stdscore[i1] < _mag_x_adds_stdscore[0])
			_mag_x_adds_stdscore[0] = _mag_x_stdscore[i1]; // min
		if(_mag_x_stdscore[i1] > _mag_x_adds_stdscore[1])
			_mag_x_adds_stdscore[1] = _mag_x_stdscore[i1]; // max
		_mag_x_adds_stdscore[4] = _mag_x_adds_stdscore[2];
		_mag_x_adds_stdscore[2] = _mag_x_adds_stdscore[2] + (_mag_x_stdscore[i1]-_mag_x_adds_stdscore[2])/i1; //avg
		_mag_x_adds_stdscore[3] = (_mag_x_stdscore[i1] - _mag_x_adds_stdscore[2])/i1;
		sqrt_tmp = 1.0;
		sqrt_tmp = (((i1 - 1.0) * _mag_x_adds_stdscore[3] * _mag_x_adds_stdscore[3] + (_mag_x_stdscore[i1] - _mag_x_adds_stdscore[4]) * (_mag_x_stdscore[i1] - _mag_x_adds_stdscore[2]))/i1);
		_mag_x_adds_stdscore[3] = sqrt(sqrt_tmp);
		// _mag_x_adds_stdscore[3] = std::sqrt(
		
		if(_mag_y_stdscore[i1] < _mag_y_adds_stdscore[0])
			_mag_y_adds_stdscore[0] = _mag_y_stdscore[i1]; // min
		if(_mag_y_stdscore[i1] > _mag_y_adds_stdscore[1])
			_mag_y_adds_stdscore[1] = _mag_y_stdscore[i1]; // max
		
		_mag_y_adds_stdscore[4] = _mag_y_adds_stdscore[2];
		_mag_y_adds_stdscore[2] = _mag_y_adds_stdscore[2] + (_mag_y_stdscore[i1]-_mag_y_adds_stdscore[2])/i1; //avg
		_mag_y_adds_stdscore[3] = sqrt(((i1 - 1.0) * _mag_y_adds_stdscore[3] * _mag_y_adds_stdscore[3] + (_mag_y_stdscore[i1] - _mag_y_adds_stdscore[4]) * (_mag_y_stdscore[i1] - _mag_y_adds_stdscore[2]))/i1);

			
		if(_mag_z_stdscore[i1] < _mag_z_adds_stdscore[0])
			_mag_z_adds_stdscore[0] = _mag_z_stdscore[i1]; // min
		if(_mag_z_stdscore[i1] > _mag_z_adds_stdscore[1])
			_mag_z_adds_stdscore[1] = _mag_z_stdscore[i1]; // max
		_mag_z_adds_stdscore[4] = _mag_z_adds_stdscore[2];
		_mag_z_adds_stdscore[2] = _mag_z_adds_stdscore[2] + (_mag_z_stdscore[i1]-_mag_z_adds_stdscore[2])/i1; //avg
		_mag_z_adds_stdscore[3] = sqrt(((i1 - 1.0) * _mag_z_adds_stdscore[3] * _mag_z_adds_stdscore[3] + (_mag_z_stdscore[i1] - _mag_z_adds_stdscore[4]) * (_mag_z_stdscore[i1] - _mag_z_adds_stdscore[2]))/i1);
			
		if(_acc_x_stdscore[i1] < _acc_x_adds_stdscore[0])
			_acc_x_adds_stdscore[0] = _acc_x_stdscore[i1]; // min
		if(_acc_x_stdscore[i1] > _acc_x_adds_stdscore[1])
			_acc_x_adds_stdscore[1] = _acc_x_stdscore[i1]; // max
		_acc_x_adds_stdscore[4] = _acc_x_adds_stdscore[2];
		_acc_x_adds_stdscore[2] = _acc_x_adds_stdscore[2] + (_acc_x_stdscore[i1]-_acc_x_adds_stdscore[2])/i1; //avg
		_acc_x_adds_stdscore[3] = sqrt(((i1 - 1.0) * _acc_x_adds_stdscore[3] * _acc_x_adds_stdscore[3] + (_acc_x_stdscore[i1] - _acc_x_adds_stdscore[4]) * (_acc_x_stdscore[i1] - _acc_x_adds_stdscore[2]))/i1);

		if(_acc_y_stdscore[i1] < _acc_y_adds_stdscore[0])
			_acc_y_adds_stdscore[0] = _acc_y_stdscore[i1]; // min
		if(_acc_y_stdscore[i1] > _acc_y_adds_stdscore[1])
			_acc_y_adds_stdscore[1] = _acc_y_stdscore[i1]; // max
		_acc_y_adds_stdscore[4] = _acc_y_adds_stdscore[2];
		_acc_y_adds_stdscore[2] = _acc_y_adds_stdscore[2] + (_acc_y_stdscore[i1]-_acc_y_adds_stdscore[2])/i1; //avg
		_acc_y_adds_stdscore[3] = sqrt(((i1 - 1.0) * _acc_y_adds_stdscore[3] * _acc_y_adds_stdscore[3] + (_acc_y_stdscore[i1] - _acc_y_adds_stdscore[4]) * (_acc_y_stdscore[i1] - _acc_y_adds_stdscore[2]))/i1);
			
		if(_acc_z_stdscore[i1] < _acc_z_adds_stdscore[0])
			_acc_z_adds_stdscore[0] = _acc_z_stdscore[i1]; // min
		if(_acc_z_stdscore[i1] > _acc_z_adds_stdscore[1])
			_acc_z_adds_stdscore[1] = _acc_z_stdscore[i1]; // max
		_acc_z_adds_stdscore[4] = _acc_z_adds_stdscore[2];
		_acc_z_adds_stdscore[2] = _acc_z_adds_stdscore[2] + (_acc_z_stdscore[i1]-_acc_z_adds_stdscore[2])/i1; //avg
		_acc_z_adds_stdscore[3] = sqrt(((i1 - 1.0) * _acc_z_adds_stdscore[3] * _acc_z_adds_stdscore[3] + (_acc_z_stdscore[i1] - _acc_z_adds_stdscore[4]) * (_acc_z_stdscore[i1] - _acc_z_adds_stdscore[2]))/i1);
		
		if(_gyr_x_stdscore[i1] < _gyr_x_adds_stdscore[0])
			_gyr_x_adds_stdscore[0] = _gyr_x_stdscore[i1]; // min
		if(_gyr_x_stdscore[i1] > _gyr_x_adds_stdscore[1])
			_gyr_x_adds_stdscore[1] = _gyr_x_stdscore[i1]; // max
		_gyr_x_adds_stdscore[4] = _gyr_x_adds_stdscore[2];
		_gyr_x_adds_stdscore[2] = _gyr_x_adds_stdscore[2] + (_gyr_x_stdscore[i1]-_gyr_x_adds_stdscore[2])/i1; //avg
		_gyr_x_adds_stdscore[3] = sqrt(((i1 - 1.0) * _gyr_x_adds_stdscore[3] * _gyr_x_adds_stdscore[3] + (_gyr_x_stdscore[i1] - _gyr_x_adds_stdscore[4]) * (_gyr_x_stdscore[i1] - _gyr_x_adds_stdscore[2]))/i1);
			
		if(_gyr_y_stdscore[i1] < _gyr_y_adds_stdscore[0])
			_gyr_y_adds_stdscore[0] = _gyr_y_stdscore[i1]; // min
		if(_gyr_y_stdscore[i1] > _gyr_y_adds_stdscore[1])
			_gyr_y_adds_stdscore[1] = _gyr_y_stdscore[i1]; // max
		_gyr_y_adds_stdscore[4] = _gyr_y_adds_stdscore[2];
		_gyr_y_adds_stdscore[2] = _gyr_y_adds_stdscore[2] + (_gyr_y_stdscore[i1]-_gyr_y_adds_stdscore[2])/i1; //avg
		_gyr_y_adds_stdscore[3] = sqrt(((i1 - 1.0) * _gyr_y_adds_stdscore[3] * _gyr_y_adds_stdscore[3] + (_gyr_y_stdscore[i1] - _gyr_y_adds_stdscore[4]) * (_gyr_y_stdscore[i1] - _gyr_y_adds_stdscore[2]))/i1);
			
		if(_gyr_z_stdscore[i1] < _gyr_z_adds_stdscore[0])
			_gyr_z_adds_stdscore[0] = _gyr_z_stdscore[i1]; // min
		if(_gyr_z_stdscore[i1] > _gyr_z_adds_stdscore[1])
			_gyr_z_adds_stdscore[1] = _gyr_z_stdscore[i1]; // max
		_gyr_z_adds_stdscore[4] = _gyr_z_adds_stdscore[2];
		_gyr_z_adds_stdscore[2] = _gyr_z_adds_stdscore[2] + (_gyr_z_stdscore[i1]-_gyr_z_adds_stdscore[2])/i1; //avg
		_gyr_z_adds_stdscore[3] = sqrt(((i1 - 1.0) * _gyr_z_adds_stdscore[3] * _gyr_z_adds_stdscore[3] + (_gyr_z_stdscore[i1] - _gyr_z_adds_stdscore[4]) * (_gyr_z_stdscore[i1] - _gyr_z_adds_stdscore[2]))/i1);
			
		if(_pitch_stdscore[i1] < _pitch_adds_stdscore[0])
			_pitch_adds_stdscore[0] = _pitch_stdscore[i1]; // min
		if(_pitch_stdscore[i1] > _pitch_adds_stdscore[1])
			_pitch_adds_stdscore[1] = _pitch_stdscore[i1]; // max
		_pitch_adds_stdscore[4] = _pitch_adds_stdscore[2];
		_pitch_adds_stdscore[2] = _pitch_adds_stdscore[2] + (_pitch_stdscore[i1]-_pitch_adds_stdscore[2])/i1; //avg
		_pitch_adds_stdscore[3] = sqrt(((i1 - 1.0) * _pitch_adds_stdscore[3] * _pitch_adds_stdscore[3] + (_pitch_stdscore[i1] - _pitch_adds_stdscore[4]) * (_pitch_stdscore[i1] - _pitch_adds_stdscore[2]))/i1);
			
		if(_roll_stdscore[i1] < _roll_adds_stdscore[0])
			_roll_adds_stdscore[0] = _roll_stdscore[i1]; // min
		if(_roll_stdscore[i1] > _roll_adds_stdscore[1])
			_roll_adds_stdscore[1] = _roll_stdscore[i1]; // max
		_roll_adds_stdscore[4] = _roll_adds_stdscore[2];
		_roll_adds_stdscore[2] = _roll_adds_stdscore[2] + (_roll_stdscore[i1]-_roll_adds_stdscore[2])/i1; //avg
		_roll_adds_stdscore[3] = sqrt(((i1 - 1.0) * _roll_adds_stdscore[3] * _roll_adds_stdscore[3] + (_roll_stdscore[i1] - _roll_adds_stdscore[4]) * (_roll_stdscore[i1] - _roll_adds_stdscore[2]))/i1);
			
		if(_yaw_stdscore[i1] < _yaw_adds_stdscore[0])
			_yaw_adds_stdscore[0] = _yaw_stdscore[i1]; // min
		if(_yaw_stdscore[i1] > _yaw_adds_stdscore[1])
			_yaw_adds_stdscore[1] = _yaw_stdscore[i1]; // max
		_yaw_adds_stdscore[4] = _yaw_adds_stdscore[2];
		_yaw_adds_stdscore[2] = _yaw_adds_stdscore[2] + (_yaw_stdscore[i1]-_yaw_adds_stdscore[2])/i1; //avg
		_yaw_adds_stdscore[3] = sqrt(((i1 - 1.0) * _yaw_adds_stdscore[3] * _yaw_adds_stdscore[3] + (_yaw_stdscore[i1] - _yaw_adds_stdscore[4]) * (_yaw_stdscore[i1] - _yaw_adds_stdscore[2]))/i1);
	}
	// normalize between -1 and 1
	for (i1 = 1; i1 < _nb_elem; ++i1)
	{
		_mag_x_stdscore_norm.push_back(-1 + (2*(_mag_x_stdscore[i1] - _mag_x_adds_stdscore[0]))/((_mag_x_adds_stdscore[1] - _mag_x_adds_stdscore[0])));
		_mag_y_stdscore_norm.push_back(-1 + (2*(_mag_y_stdscore[i1] - _mag_y_adds_stdscore[0]))/((_mag_y_adds_stdscore[1] - _mag_y_adds_stdscore[0])));
		_mag_z_stdscore_norm.push_back(-1 + (2*(_mag_z_stdscore[i1] - _mag_z_adds_stdscore[0]))/((_mag_z_adds_stdscore[1] - _mag_z_adds_stdscore[0])));
		
		_acc_x_stdscore_norm.push_back(-1 + (2*(_acc_x_stdscore[i1] - _acc_x_adds_stdscore[0]))/((_acc_x_adds_stdscore[1] - _acc_x_adds_stdscore[0])));
		_acc_y_stdscore_norm.push_back(-1 + (2*(_acc_y_stdscore[i1] - _acc_y_adds_stdscore[0]))/((_acc_y_adds_stdscore[1] - _acc_y_adds_stdscore[0])));
		_acc_z_stdscore_norm.push_back(-1 + (2*(_acc_z_stdscore[i1] - _acc_z_adds_stdscore[0]))/((_acc_z_adds_stdscore[1] - _acc_z_adds_stdscore[0])));
		
		_gyr_x_stdscore_norm.push_back(-1 + (2*(_gyr_x_stdscore[i1] - _gyr_x_adds_stdscore[0]))/((_gyr_x_adds_stdscore[1] - _gyr_x_adds_stdscore[0])));
		_gyr_y_stdscore_norm.push_back(-1 + (2*(_gyr_y_stdscore[i1] - _gyr_y_adds_stdscore[0]))/((_gyr_y_adds_stdscore[1] - _gyr_y_adds_stdscore[0])));
		_gyr_z_stdscore_norm.push_back(-1 + (2*(_gyr_z_stdscore[i1] - _gyr_z_adds_stdscore[0]))/((_gyr_z_adds_stdscore[1] - _gyr_z_adds_stdscore[0])));
		
		_pitch_stdscore_norm.push_back(-1 + (2*(_pitch_stdscore[i1] - _pitch_adds_stdscore[0]))/((_pitch_adds_stdscore[1] - _pitch_adds_stdscore[0])));
		_roll_stdscore_norm.push_back(-1 + (2*(_roll_stdscore[i1] - _roll_adds_stdscore[0]))/((_roll_adds_stdscore[1] - _roll_adds_stdscore[0])));
		_yaw_stdscore_norm.push_back(-1 + (2*(_yaw_stdscore[i1] - _yaw_adds_stdscore[0]))/((_yaw_adds_stdscore[1] - _yaw_adds_stdscore[0])));
		
	}
}
//---------------------------------------------------------------------
void px4_log_handler::randomize(){
	std::random_shuffle(_random_it_vector_train.begin(), _random_it_vector_train.end());
}
//---------------------------------------------------------------------
void px4_log_handler::create_random_dataset(uint32_t sample_nb, uint32_t nb_timestep)
{
	uint32_t i1, i2, split;
	_max_train_samples = 0;
	_max_test_samples = 0;
	_random_it_vector_train.clear();
	for (i1 = 0; i1 < (_nb_elem-nb_timestep-sample_nb-1); ++i1)
	{
		_random_it_vector_train.push_back(i1);
	}

	std::random_shuffle(_random_it_vector_train.begin(), _random_it_vector_train.end());

	split = round(_random_it_vector_train.size()*0.3);

	std::vector<uint32_t>::const_iterator first = _random_it_vector_train.begin();
	std::vector<uint32_t>::const_iterator last = _random_it_vector_train.begin() + split;
	std::vector<uint32_t> test_batches(first, last);
	_random_it_vector_test = test_batches;
	
	_random_it_vector_train.erase(first, last);

	_max_train_samples = _random_it_vector_train.size()-1;
	_max_test_samples = _random_it_vector_test.size()-1;
}



//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> px4_log_handler::get_input_rdm_train(uint32_t sample_nb, uint32_t nb_timestep, uint32_t input_size)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> input;
	for(i1 = 0; i1 < sample_nb; ++i1) 
	{
		std::vector<std::vector<float>> input_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> input_tmp_i3;
			index_tmp = _random_it_vector_train[i1] + i2;
			//std::cout << "index_tmp = " << index_tmp << std::endl;
			input_tmp_i3.push_back(_mag_x_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_mag_y_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_mag_z_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_acc_x_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_acc_y_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_acc_z_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_gyr_x_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_gyr_y_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_gyr_z_stdscore_norm[index_tmp]);
			input_tmp_i2.push_back(input_tmp_i3);
		}
		input.push_back(input_tmp_i2);
	}
	
	return input;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> px4_log_handler::get_target_rdm_train(uint32_t sample_nb, uint32_t nb_timestep, uint32_t output_size)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> target;
	for(i1 = 0; i1 < sample_nb; ++i1) 
	{
		std::vector<std::vector<float>> target_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> target_tmp_i3;
			index_tmp = _random_it_vector_train[i1] + i2;
			target_tmp_i3.push_back(_pitch_stdscore_norm[index_tmp]);
			target_tmp_i3.push_back(_roll_stdscore_norm[index_tmp]);
			target_tmp_i3.push_back(_yaw_stdscore_norm[index_tmp]);
			target_tmp_i2.push_back(target_tmp_i3);
		}
		target.push_back(target_tmp_i2);
	}
	
	return target;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> px4_log_handler::get_input_rdm_test(uint32_t sample_nb, uint32_t nb_timestep, uint32_t input_size)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> input;
	for(i1 = 0; i1 < sample_nb; ++i1) 
	{
		std::vector<std::vector<float>> input_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> input_tmp_i3;
			index_tmp = _random_it_vector_test[i1] + i2;
			//std::cout << "index_tmp = " << index_tmp << std::endl;
			input_tmp_i3.push_back(_mag_x_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_mag_y_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_mag_z_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_acc_x_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_acc_y_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_acc_z_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_gyr_x_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_gyr_y_stdscore_norm[index_tmp]);
			input_tmp_i3.push_back(_gyr_z_stdscore_norm[index_tmp]);
			input_tmp_i2.push_back(input_tmp_i3);
		}
		input.push_back(input_tmp_i2);
	}
	
	return input;
}
//---------------------------------------------------------------------
std::vector<std::vector<float>> px4_log_handler::get_input_for_inference(uint32_t sample_nb, uint32_t nb_timestep, uint32_t input_size)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<float>> input_tmp_i2;
	for(i2 = 0; i2 < _nb_elem; ++i2) 
	{
		std::vector<float> input_tmp_i3;
		index_tmp = i2;
		//std::cout << "index_tmp = " << index_tmp << std::endl;
		input_tmp_i3.push_back(_mag_x_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_mag_y_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_mag_z_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_acc_x_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_acc_y_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_acc_z_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_gyr_x_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_gyr_y_stdscore_norm[index_tmp]);
		input_tmp_i3.push_back(_gyr_z_stdscore_norm[index_tmp]);
		input_tmp_i2.push_back(input_tmp_i3);
	}
	
	return input_tmp_i2;
}
//---------------------------------------------------------------------
std::vector<std::vector<float>> px4_log_handler::get_target_for_inference(uint32_t sample_nb, uint32_t nb_timestep, uint32_t output_size)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<float>> target_tmp_i2;
	for(i2 = 0; i2 < _nb_elem; ++i2) 
	{
		std::vector<float> target_tmp_i3;
		index_tmp = i2;
		target_tmp_i3.push_back(_pitch_stdscore_norm[index_tmp]);
		target_tmp_i3.push_back(_roll_stdscore_norm[index_tmp]);
		target_tmp_i3.push_back(_yaw_stdscore_norm[index_tmp]);
		target_tmp_i2.push_back(target_tmp_i3);
	}
	
	return target_tmp_i2;
}

//---------------------------------------------------------------------
std::vector<std::vector<std::vector<float>>> px4_log_handler::get_target_rdm_test(uint32_t sample_nb, uint32_t nb_timestep, uint32_t output_size)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<std::vector<float>>> target;
	for(i1 = 0; i1 < sample_nb; ++i1) 
	{
		std::vector<std::vector<float>> target_tmp_i2;
		for(i2 = 0; i2 < nb_timestep; ++i2) 
		{
			std::vector<float> target_tmp_i3;
			index_tmp = _random_it_vector_test[i1] + i2;
			target_tmp_i3.push_back(_pitch_stdscore_norm[index_tmp]);
			target_tmp_i3.push_back(_roll_stdscore_norm[index_tmp]);
			target_tmp_i3.push_back(_yaw_stdscore_norm[index_tmp]);
			target_tmp_i2.push_back(target_tmp_i3);
		}
		target.push_back(target_tmp_i2);
	}
	
	return target;
}

//---------------------------------------------------------------------
std::vector<std::vector<float>> px4_log_handler::get_input_test(uint32_t batch, uint32_t nb_elem)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;

	std::vector<std::vector<float>> input;
	uint32_t begin =  _random_it_vector_test[batch];
	for(i2 = begin; i2 < begin + nb_elem; ++i2) 
	{
		std::vector<float> input_tmp_i3;
		//std::cout << "index_tmp = " << index_tmp << std::endl;
		input_tmp_i3.push_back(_mag_x_stdscore_norm[i2]);
		input_tmp_i3.push_back(_mag_y_stdscore_norm[i2]);
		input_tmp_i3.push_back(_mag_z_stdscore_norm[i2]);
		input_tmp_i3.push_back(_acc_x_stdscore_norm[i2]);
		input_tmp_i3.push_back(_acc_y_stdscore_norm[i2]);
		input_tmp_i3.push_back(_acc_z_stdscore_norm[i2]);
		input_tmp_i3.push_back(_gyr_x_stdscore_norm[i2]);
		input_tmp_i3.push_back(_gyr_y_stdscore_norm[i2]);
		input_tmp_i3.push_back(_gyr_z_stdscore_norm[i2]);
		input.push_back(input_tmp_i3);

	}
	
	return input;
}

//---------------------------------------------------------------------
std::vector<std::vector<float>> px4_log_handler::get_target_test(uint32_t batch, uint nb_elem)
{
	unsigned int i1,i2,i3;
	unsigned int index_tmp;
	std::vector<std::vector<float>> target;

	uint32_t begin =  _random_it_vector_test[batch];
	for(i2 = begin; i2 < begin + nb_elem; ++i2) 
	{
		std::vector<float> target_tmp_i3;

		target_tmp_i3.push_back(_pitch_stdscore_norm[i2]);
		target_tmp_i3.push_back(_roll_stdscore_norm[i2]);
		target_tmp_i3.push_back(_yaw_stdscore_norm[i2]);
		target.push_back(target_tmp_i3);
	}

	return target;
}


//---------------------------------------------------------------------
void px4_log_handler::create_report(const char *reportfilename)
{
	unsigned int i1,i2;
	std::ofstream reportfile;
	reportfile.open(reportfilename);
	std::string ite_name = "sample_iter,";
	std::string mag_name = "mag_x,mag_y,mag_z,mag_x_stdscore,mag_y_stdscore,mag_z_stdscore,mag_x_stdscore_norm,mag_y_stdscore_norm,mag_z_stdscore_norm,";
	std::string acc_name = "acc_x,acc_y,acc_z,acc_x_stdscore,acc_y_stdscore,acc_z_stdscore,acc_x_stdscore_norm,acc_y_stdscore_norm,acc_z_stdscore_norm,";
	std::string gyr_name = "gyr_x,gyr_y,gyr_z,gyr_x_stdscore,gyr_y_stdscore,gyr_z_stdscore,gyr_x_stdscore_norm,gyr_y_stdscore_norm,gyr_z_stdscore_norm,";
	std::string pry_name = "pitch,roll,yaw,pitch_stdscore,roll_stdscore,yaw_stdscore,pitch_stdscore_norm,roll_stdscore_norm,yaw_stdscore_norm";
	std::string end_of_line = "\n";
	reportfile << ite_name + mag_name + acc_name + gyr_name + pry_name + end_of_line;
	for(i1 = 1; i1 < _nb_elem; ++i1) 
	{
		reportfile << i1 << ","  << _mag_x[i1] << "," << _mag_y[i1] << "," << _mag_z[i1] << ","
		<< _mag_x_stdscore[i1] << "," << _mag_y_stdscore[i1] << "," << _mag_z_stdscore[i1] << ","
		<< _mag_x_stdscore_norm[i1] << "," << _mag_y_stdscore_norm[i1] << "," << _mag_z_stdscore_norm[i1] << ","
		
		<< _acc_x[i1] << "," << _acc_y[i1] << "," << _acc_z[i1] << ","
		<< _acc_x_stdscore[i1] << "," << _acc_y_stdscore[i1] << "," << _acc_z_stdscore[i1] << ","
		<< _acc_x_stdscore_norm[i1] << "," << _acc_y_stdscore_norm[i1] << "," << _acc_z_stdscore_norm[i1] << ","
		
		<< _gyr_x[i1] << "," << _gyr_y[i1] << "," << _gyr_z[i1] << ","
		<< _gyr_x_stdscore[i1] << "," << _gyr_y_stdscore[i1] << "," << _gyr_z_stdscore[i1] << ","
		<< _gyr_x_stdscore_norm[i1] << "," << _gyr_y_stdscore_norm[i1] << "," << _gyr_z_stdscore_norm[i1] << ","
		
		<< _pitch[i1] << "," << _roll[i1] << "," << _yaw[i1] << ","
		<< _pitch_stdscore[i1] << "," << _roll_stdscore[i1] << "," << _yaw_stdscore[i1] << ","
		<< _pitch_stdscore_norm[i1] << "," << _roll_stdscore_norm[i1] << "," << _yaw_stdscore_norm[i1] << "\n";
	}
	reportfile.close();
}
