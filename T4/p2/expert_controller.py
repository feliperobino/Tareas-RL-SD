from collections import namedtuple
import pandas as pd
import numpy as np

Action = namedtuple('Action', ['basal', 'bolus'])

class BBController:
    def __init__(self, target, patient_params, Ts=3):
        self.params = patient_params
        self.target = target
        self.Ts = Ts
        self.quest = pd.read_csv('controller_params.csv')


    def policy(self, observation, **kwargs):
        sample_time = kwargs.get('sample_time', 1)
        pname = kwargs.get('patient_name')
        meal = kwargs.get('meal')
        action = self._bb_policy(pname, meal, observation.CGM, sample_time)
        return action

    def _bb_policy(self, name, meal, glucose, env_sample_time):
        """
        Helper function to compute the basal and bolus amount.

        The basal insulin is based on the insulin amount to keep the blood
        glucose in the steady state when there is no (meal) disturbance.
               basal = u2ss (pmol/(L*kg)) * body_weight (kg) / 6000 (U/min)

        The bolus amount is computed based on the current glucose level, the
        target glucose level, the patient's correction factor and the patient's
        carbohydrate ratio.
               bolus = ((carbohydrate / carbohydrate_ratio) +
                       (current_glucose - target_glucose) / correction_factor)
                       / sample_time
        NOTE the bolus computed from the above formula is in unit U. The
        simulator only accepts insulin rate. Hence the bolus is converted to
        insulin rate.
        """

        u2ss = self.params.u2ss  # unit: pmol/(L*kg)
        BW = self.params.BW # unit: kg
        basal = u2ss * BW / 6000  # unit: U/min
        quest = self.quest[self.quest.Name.str.match(name)]
        if meal > 0:
            bolus = (
                    (meal * env_sample_time) / quest.CR.values + (glucose > 150) *
                    (glucose - self.target) / quest.CF.values).item()  # unit: U
        else:
            bolus = 0  # unit: U


        bolus = bolus / env_sample_time  # unit: U/min

        # 30% error of the Bolus
        bolus = bolus * np.random.uniform(low=0.65, high=1.35)

        return basal, bolus

    def reset(self):
        pass
