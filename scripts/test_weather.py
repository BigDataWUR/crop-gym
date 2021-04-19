import os, sys
import pcse
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np


data_dir = '../gym-crop/gym_crop/envs/env_data/'
crop = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "crop", "lintul3_winterwheat.crop"))
soil = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "soil", "lintul3_springwheat.soil"))
site = pcse.fileinput.PCSEFileReader(os.path.join(data_dir, "site", "lintul3_springwheat.site"))
parameterprovider = pcse.base.ParameterProvider(soildata=soil, cropdata=crop, sitedata=site)


with open(os.path.join(data_dir, 'agro', 'agromanagement_reactive.yaml')) as file:
    agromanagement = yaml.load(file, Loader=yaml.SafeLoader)

def replace_year(agromanagement, target_year):
    dict_ = agromanagement[0]
    old_date = next(iter(dict_.keys()))
    new_date = old_date.replace(target_year)
    content = dict_[old_date]
    crop_start_date = content['CropCalendar']['crop_start_date'].replace(target_year)
    content['CropCalendar']['crop_start_date'] = crop_start_date
    crop_end_date = content['CropCalendar']['crop_end_date'].replace(target_year+1)
    content['CropCalendar']['crop_end_date'] = crop_end_date
    dict_[new_date] = dict_.pop(old_date)
    # print(agromanagement)
    agromanagement[1] = {crop_end_date:None}
    return agromanagement

coordinates = [(x,y) for x in [51.5, 52, 52.5] for y in [5, 5.5, 6, 6.5]]
complete_years = dict()
for latitude, longitude in coordinates:
    complete_years[(latitude, longitude)] = []
    print(latitude, longitude)
    for year in np.linspace(1983, 2018, 36).astype(int):
        weatherdataprovider = pcse.db.NASAPowerWeatherDataProvider(latitude, longitude)
        agromanagement = replace_year(agromanagement, year)
        model = pcse.models.LINTUL3(parameterprovider, weatherdataprovider, agromanagement)
        try:
            model.run_till_terminate()
            complete_years[(latitude, longitude)].append(year)
        except:
            print(f'failed {latitude}, {longitude}, {year}')

print(complete_years)
