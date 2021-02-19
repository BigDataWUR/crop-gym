import datetime
import gym
import os
import yaml
import copy
import numpy as np
import pandas as pd
import pcse

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'env_data/')

class FertilizationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_dir=data_dir, intervention_interval=7, weather_forecast_length=7):
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(81,))
        crop = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "crop", "lintul3_springwheat.crop"))
        soil = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "soil", "lintul3_springwheat.soil"))
        site = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "site", "lintul3_springwheat.site"))
        self.parameterprovider = pcse.base.ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)
        self.weatherdataprovider = pcse.db.NASAPowerWeatherDataProvider(52, 5.2)
        self.agromanagement, self.crop_end_date = self._load_agromanagement_data()
        self.model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider, self.agromanagement)
        self.intervention_interval = intervention_interval
        self.weather_forecast_length = weather_forecast_length

    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        output = self._run_simulation()
        observation = self._process_output(output)

        growth = output['WSO'][-1] - output['WSO'][-1-self.intervention_interval]
        reward = growth if not np.isnan(growth) else 0

        done = self.date > self.crop_end_date
        return observation, reward, done, {}

    def _load_agromanagement_data(self):
        agromanagement = yaml.safe_load(open(os.path.join(data_dir, "agro/wofost_npk.agro")))['AgroManagement']
        crop_end_date = list(agromanagement[0].values())[0]['CropCalendar']['crop_end_date']
        return agromanagement, crop_end_date

    def _process_output(self, output): 
        crop_observation = np.array(output.iloc[-1])
        # forecast for the week after the observation
        weather_forecast = get_weather(self.weatherdataprovider, self.date, self.weather_forecast_length)
        observation = np.concatenate([crop_observation, weather_forecast.flatten()])
        observation = np.nan_to_num(observation)
        return observation

    def _run_simulation(self):
        self.model.run(days=self.intervention_interval)
        output = pd.DataFrame(self.model.get_output()).set_index("day")
        output = output.fillna(value=np.nan)
        self.date = output.index[-1]
        return output

    def _take_action(self, action):
        self.model._send_signal(signal=pcse.signals.irrigate, amount=2*action, efficiency=0.7) # water in cm

    def reset(self):
        self.model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider, self.agromanagement)
        output = self._run_simulation()
        observation = self._process_output(output)
        return observation

    def render(self, mode='human', close=False):
        pass

def get_weather(weatherdataprovider, date, days):
    dates = [date + datetime.timedelta(i) for i in range(0, days)]
    weather = [get_weather_day(weatherdataprovider, day) for day in dates]
    return np.array(weather)

def get_weather_day(weatherdataprovider, date):
    weatherdatacontainer = weatherdataprovider(date)
    weather = [getattr(weatherdatacontainer, attr) for attr in weatherdatacontainer.required]
    return weather
