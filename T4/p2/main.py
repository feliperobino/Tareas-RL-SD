from hovorka import HovorkaModel, sim_to_hov_u, cgm_to_mmol, MGDL_TO_MMOL, MMOL_TO_MGDL, G_TO_MMOL
from KF import DiscreteKalmanFilter
import numpy as np
from simglucose.simulation.scenario_gen import RandomScenario
from datetime import datetime
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
import matplotlib.pyplot as plt
from simglucose.patient.t1dpatient import T1DPatient, Action
from simglucose.simulation.env import T1DSimEnv
from expert_controller import BBController
from collections import namedtuple


Action = namedtuple('Action', ['basal', 'bolus'])


def get_hovorka_matrices(patient_name, Ts=1, sensitivity_scale=1):
    patient = T1DPatient.withName(patient_name)
    params  = patient._params

    # Basal information which is known in practice
    y_ss_mgdL = float(patient.init_state[12] / params.Vg)
    u_basal = params.u2ss * params.BW / 6000 * 60
    print(f"Patient {patient_name}: equilibrium BG = {y_ss_mgdL:.1f} mg/dL")
    BW = float(params.BW)


    # Hovorka functions
    model = HovorkaModel(Ts=Ts, BW=BW, SI_scale=sensitivity_scale)
    _, x0 = model.get_basal(G=y_ss_mgdL * MGDL_TO_MMOL)

    model.state  = x0
    Ad, Bd, Cd, Dd, dt = model.get_linear_system(x_eq=x0, u_eq=[u_basal, 0])


    u_eq_internal = np.array(model.u_eq, dtype=float)
    y_eq          = float(x0[10])
    return Ad, Bd, Cd, Dd, dt, x0, u_eq_internal, y_eq



def sim_together(patient_name, sensitivity_scale):
    now     = datetime.now()
    patient = T1DPatient.withName(patient_name)
    params = patient._params
    y_ss_mgdL = float(patient.init_state[12] / params.Vg)
    sensor  = CGMSensor.withName('Dexcom', seed=0)
    pump    = InsulinPump.withName('Insulet')
    scenario = RandomScenario(start_time=now)
    env     = T1DSimEnv(patient, sensor, pump, scenario)

    observation, reward, done, info = env.reset()
    Ts = int(info['sample_time'])

    controller = BBController(target=y_ss_mgdL, patient_params=params, Ts=Ts)

    # Hovorka Linear Matrices
    Ad, Bd, Cd, Dd, dt, x0, u_eq, y_eq = get_hovorka_matrices(patient_name, Ts=Ts,
                                                              sensitivity_scale=sensitivity_scale)

    # Kalman to estimate the state
    n  = Ad.shape[0]
    Q  = np.eye(n) * 1e-5   # process noise
    R  = np.array([[1e-1]])  # measurement noise
    P0 = np.eye(n) * 10
    kf_hov = DiscreteKalmanFilter(
        Ad, Bd, Cd, Dd, Q, R, P0,
        x_eq=x0,
        u_eq=u_eq,
        y_eq=[y_eq],
        x_init=np.zeros((n, 1)),
    )

    day_minutes = int(24 * 60 / Ts)
    print(f"Total samples: {day_minutes}")
    t    = 0
    done = False

    observation_list  = []
    glucose_list      = []
    meal_list         = []
    insulin_list      = []
    reward_list       = []
    glucose_est_hov_list  = []
    glucose_open_loop = []
    x_est_list = []

    x_ol = np.zeros_like(x0)

    while not done:
        ################# Simglucose #######################
        u_basal, u_bolus = controller.policy(observation, **info) # Expert Controller with error

        ################# Controller #####################
        # Your controller should only modify u_basal
        rl_modifier = 5*np.random.rand() # This is just an option it can be additive as well
        u_basal = u_basal * rl_modifier

        action = Action(u_basal, u_bolus)
        observation, reward, terminated, info = env.step(action)
        done = terminated
        t   += 1

        ################# Hovorka #################################
        # Convert from the simglucose units to the Hovorka Units
        insulin_U_min  = action.basal + action.bolus   # U/min
        meal_g_per_min = info['meal']                  # g/min
        cgm_mmol       = cgm_to_mmol(observation.CGM) # mmol/L
        u_hov  = sim_to_hov_u(insulin_U_min, meal_g_per_min)  # [mU/min, mmol/min]



        # Kalman State estimation
        x_est  = kf_hov.step(u=u_hov, y=cgm_mmol)
        g_est_mmol = float((Cd @ x_est)[0])
        g_hov_mgdL = g_est_mmol * MMOL_TO_MGDL

        # Open Loop Prediction
        x_ol = Ad @ x_ol.reshape(-1, 1) + Bd @ (u_hov - u_eq).reshape(-1, 1)
        g_ol = Cd @ x_ol + y_eq
        g_ol_mgdL = (g_ol* MMOL_TO_MGDL).squeeze()


        observation_list.append(observation.CGM)
        glucose_list.append(float(info['bg']))
        meal_list.append(info['meal'])
        insulin_list.append(insulin_U_min)
        reward_list.append(reward) # Risk could also be used as the reward
        glucose_est_hov_list.append(g_hov_mgdL)
        glucose_open_loop.append(g_ol_mgdL)
        x_est_list.append(x_est.reshape(1, -1))

        if done or t >= day_minutes:
            break

    x_est_list = np.concatenate(x_est_list, axis=0)

    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(observation_list,   label='CGM [mg/dL]')
    plt.plot(glucose_list,       label='BG [mg/dL]')
    plt.plot(glucose_est_hov_list,   label='KF C-state [mg/dL]', linestyle='--')
    plt.plot(glucose_open_loop,   label='Open Loop', linestyle='--')

    plt.ylabel('Glucose')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(reward_list)
    plt.ylabel('Risk')
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(meal_list)
    plt.ylabel('Meal [g/min]')
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(insulin_list)
    plt.ylabel('Insulin [U/min]')
    plt.grid()

    plt.tight_layout()


    plt.figure(figsize=(15, 8))
    plt.suptitle('Estimated states')
    for i in range(x_est_list.shape[1]):
        plt.subplot(3, 4, i + 1)
        plt.plot(x_est_list[:, i], label='x{}'.format(i))
        plt.grid()
        plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    selected_adult = "adult#001"
    scale = {"adult#001":0.35, "adult#002": 0.35, "adult#003":0.35, "adult#004":0.45,
             "adult#005":0.28, "adult#006":0.28, "adult#007":0.6, "adult#008":0.45,
             "adult#009": 0.21,  "adult#010":0.23}

    sim_together(patient_name=selected_adult, sensitivity_scale=scale[selected_adult])
