import datetime
import gym
import os
import yaml
import copy
import numpy as np
import pandas as pd
import pcse

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'env_data/')

all_years = range(1983, 2018)
missing_data = [2007, 2008, 2010, 2013, 2015, 2017]
test_years = [1984, 1994, 2004, 2014]
train_weather_data = [year for year in all_years if year not in missing_data+test_years]

class FertilizationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_dir=data_dir, intervention_interval=7, weather_forecast_length=7, beta=1, seed=0, fixed_year=None, fixed_location=None):
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(81,))
        crop = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "crop", "lintul3_winterwheat.crop"))
        soil = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "soil", "lintul3_springwheat.soil"))
        site = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "site", "lintul3_springwheat.site"))
        self.parameterprovider = pcse.base.ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)
        self.intervention_interval = intervention_interval
        self.weather_forecast_length = weather_forecast_length
        self.beta = beta
        self.amount = 0.025*self.intervention_interval
        self.seed(seed)
        self.fixed_location = fixed_location
        self.weatherdataprovider = self._get_weatherdataprovider()
        self.fixed_year = fixed_year
        self.agromanagement = self._load_agromanagement_data()
        self.model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider, self.agromanagement)
        self.baseline_model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider, self.agromanagement)
        self.log = self._init_log()

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
        fertilizer = self._take_action(action)
        output = self._run_simulation(self.model)
        baseline_output = self._run_simulation(self.baseline_model)
        self.date = output.index[-1]
        observation = self._process_output(output)

        growth = output['WSO'][-1] - output['WSO'][-1-self.intervention_interval]
        growth = growth if not np.isnan(growth) else 0
        baseline_growth = baseline_output['WSO'][-1] - baseline_output['WSO'][-1-self.intervention_interval]
        baseline_growth = baseline_growth if not np.isnan(baseline_growth) else 0

        reward = growth - baseline_growth - self.beta * fertilizer
        done = self.date >= self.crop_end_date

        self._log(growth, baseline_growth, fertilizer, reward)

        info = {**output.to_dict(), **self.log}

        return observation, reward, done, info

    def _init_log(self):
        return {'growth': dict(), 'baseline_growth': dict(), 'fertilizer': dict(), 'reward': dict()}

    def _load_agromanagement_data(self):
        with open(os.path.join(data_dir, 'agro/agromanagement_irrigation.yaml')) as file:
            agromanagement = yaml.load(file, Loader=yaml.SafeLoader)
        self._replace_year(agromanagement)
        return agromanagement

    def _log(self, growth, baseline_growth, fertilizer, reward):
        self.log['growth'][self.date] = growth
        self.log['baseline_growth'][self.date] = baseline_growth
        self.log['fertilizer'][self.date - datetime.timedelta(self.intervention_interval)] = fertilizer
        self.log['reward'][self.date] = reward

    def _process_output(self, output): 
        crop_observation = np.array(output.iloc[-1])
        # forecast for the week after the observation
        weather_forecast = get_weather(self.weatherdataprovider, self.date, self.weather_forecast_length)
        observation = np.concatenate([crop_observation, weather_forecast.flatten()])
        observation = np.nan_to_num(observation)
        return observation

    def _replace_year(self, agromanagement):
        dict_ = agromanagement[0]
        old_date = next(iter(dict_.keys()))
        target_year = self.np_random.choice(train_weather_data) if not self.fixed_year else self.fixed_year
        new_date = old_date.replace(target_year)
        content = dict_[old_date]
        self.crop_start_date = content['CropCalendar']['crop_start_date'].replace(target_year)
        content['CropCalendar']['crop_start_date'] = self.crop_start_date
        self.crop_end_date = content['CropCalendar']['crop_end_date'].replace(target_year+1)
        content['CropCalendar']['crop_end_date'] = self.crop_end_date
        dict_[new_date] = dict_.pop(old_date)
        return agromanagement

    def _get_weatherdataprovider(self):
        location = self.fixed_location
        if not location:
            latitude = self.np_random.choice([51.5, 52, 52.5])
            longitude = self.np_random.choice([5, 5.5, 6])
            location = (latitude, longitude)
        return pcse.db.NASAPowerWeatherDataProvider(*location)

    def _run_simulation(self, model):
        model.run(days=self.intervention_interval)
        output = pd.DataFrame(model.get_output()).set_index("day")
        output = output.fillna(value=np.nan)
        return output

    def _take_action(self, action):
        amount = action**2*self.amount # in g/m^2
        self.model._send_signal(signal=pcse.signals.apply_n, amount=amount, recovery=0.7)
        return amount

    def reset(self):
        self.log = self._init_log()
        self._replace_year(self.agromanagement)
        self.weatherdataprovider = self._get_weatherdataprovider()
        self.crop_start_date = list(self.agromanagement[0].values())[0]['CropCalendar']['crop_start_date']
        self.crop_end_date = list(self.agromanagement[0].values())[0]['CropCalendar']['crop_end_date']
        self.date = self.crop_start_date
        self.model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider, self.agromanagement)
        self.baseline_model = pcse.models.LINTUL3(self.parameterprovider, self.weatherdataprovider, self.agromanagement)
        self.growth = []
        self.baseline_growth = []
        output = self._run_simulation(self.model)
        baseline_output = self._run_simulation(self.baseline_model)
        observation = self._process_output(output)
        return observation

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


def get_weather(weatherdataprovider, date, days):
    dates = [date + datetime.timedelta(i) for i in range(0, days)]
    weather = [get_weather_day(weatherdataprovider, day) for day in dates]
    return np.array(weather)

def get_weather_day(weatherdataprovider, date):
    weatherdatacontainer = weatherdataprovider(date)
    weather = [getattr(weatherdatacontainer, attr) for attr in weatherdatacontainer.required]
    return weather
