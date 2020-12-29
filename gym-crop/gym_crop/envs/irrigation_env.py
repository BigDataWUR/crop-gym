import datetime
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import yaml
import copy

import numpy as np
import pandas as pd

from pcse.fileinput import ExcelWeatherDataProvider
from pcse.models import Wofost71_WLP_FD
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.util import WOFOST71SiteDataProvider
from pcse.base import ParameterProvider

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'env_data/')

class IrrigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(74,))
        self.past_actions = []
        crop = YAMLCropDataProvider()
        soil = CABOFileReader(os.path.join(data_dir, "soil", "ec3.soil"))
        site = WOFOST71SiteDataProvider(WAV=100,CO2=360)
        self.parameterprovider = ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)
        weatherfile = os.path.join(data_dir, 'meteo', 'nl1.xlsx')
        self.weatherdataprovider = ExcelWeatherDataProvider(weatherfile)
        self.agromanagement, crop_start_date = self._load_agromanagement_data()
        self.dates = [crop_start_date]

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
        self.past_actions.append(action)
        # observation takes place one week after the chosen action
        self.dates.append(self.dates[-1] + datetime.timedelta(7))
        agromanagement = self._create_agromanagement_file()
        wofost = Wofost71_WLP_FD(self.parameterprovider, self.weatherdataprovider, agromanagement)
        wofost.run_till_terminate()
        output = pd.DataFrame(wofost.get_output()).set_index("day")
        crop_observation = np.array(output.iloc[-1])
        # forecast for the week after the observation
        weather_forecast = get_weather(self.weatherdataprovider, self.dates[-1], 7)
        observation = np.concatenate([crop_observation, weather_forecast.flatten()])

        reward = output['TWSO'][-1] if not np.isnan(output['TWSO'][-1]) else 0

        return observation, reward, wofost.flag_terminate, {yaml.dump(agromanagement)}

    def _create_agromanagement_file(self):
        event_table = []
        for date, action in zip(self.dates[:-1], self.past_actions):
            if action == 1:
                event_table.append({date: {'amount': 10, 'efficiency': 0.7}})
        new_management = copy.deepcopy(self.agromanagement)
        next(iter(new_management[0].values()))['TimedEvents'][0]['events_table'] = event_table
        # terminate on the observation date
        new_management.append({self.dates[-1]: None})

        return new_management

    def _load_agromanagement_data(self):
        with open(os.path.join(data_dir, 'agro/agromanagement_irrigation.yaml')) as file:
            agromanagement = yaml.load(file, Loader=yaml.SafeLoader)
        crop_start_date = list(agromanagement[0].values())[0]['CropCalendar']['crop_start_date']
        return agromanagement, crop_start_date


def get_weather(weatherdataprovider, date, days):
    dates = [date + datetime.timedelta(i) for i in range(0, days)]
    weather = [get_weather_day(weatherdataprovider, day) for day in dates]
    return np.array(weather)

def get_weather_day(weatherdataprovider, date):
    weatherdatacontainer = weatherdataprovider(date)
    weather = [getattr(weatherdatacontainer, attr) for attr in weatherdatacontainer.required]
    return weather
