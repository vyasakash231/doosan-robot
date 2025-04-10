// main.cpp
CDRFLEx drfl;
drfl.connect_rt_control();
string version = “v1.0”;
float period = 0.001; // 1 msec
int losscount = 4;
Drfl.set_on_rt_monitoring_data(OnRTMonitoringData);
drfl.set_rt_control_output(version, period, losscount);
drfl.start_rt_control();

float time = 0.0;
int count = 0;
float q[NUMBER_OF_JOINT] = {0.0, };
float q_dot[NUMBER_OF_JOINT] = {0.0, };
float q_d[NUMBER_OF_JOINT] = {0.0, };
float q_dot_d[NUMBER_OF_JOINT] = {0.0, };
float vel_d[NUMBER_OF_JOINT] = {0.0, };
float acc_d[NUMBER_OF_JOINT] = {0.0, };
float kv[NUMBER_OF_JOINT] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // have to tune
float kp[NUMBER_OF_JOINT] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // have to tune
float ki[NUMBER_OF_JOINT] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // have to tune
float integral_v_error[NUMBER_OF_JOINT] = {0.0, };
while (1)
{
    time=(++count)*period;
    // get current state
    memcpy(q, drfl.read_data_rt()->actual_joint_position, sizeof(float)*6);
    memcpy(q_dot, drfl.read_data_rt()->actual_joint_velocity, sizeof(float)*6);

    // make trajectory
    TrajectoryGenerator(q_d, q_dot_d); // Custom Trajectory Generation Function

    // velocity feedforward + pi controller
    for (int i=0; i<6; i++) {
        q_dot_v[i] = kv[i] * (q_d[i] - q[i]);
        integral_v_error[i] += q_dot_v[i] - q_dot[i];
        vel_d[i] = q_dot_d[i] + kp[I;]*(q_dot_v[i] – q_dot[i]) + ki[i]*integral_v_error[i];
        acc_d[i] = -10000;
    }

    drfl.speedj_rt(vel_d, acc_d[i], period);

    if(time > plan1.time)
    {
        time=0;
        Drfl.stop(STOP_TYPE_SLOW);
        break;
    }

    rt_task_wait_period(NULL); // RTOS function
}