import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from src.utils.functions import mkdir_if_not_exists
from src.utils.logger import log
from logging import INFO

class ParticipantData:
    @classmethod
    def catch_data(cls, _id: int, path: str = f'data/pecanstreet/aggregate/'):
        return {"_id": _id, "data": pd.read_csv(f"{path}/{str(_id)}.csv")}

    @staticmethod
    def init_weather_readings(path: str = 'data/pecanstreet/') -> pd.DataFrame:
        try:
            weather_df = pd.read_csv(f"{path}/weather_data/162.89.0.47.csv")
        except:
            raise FileExistsError(
                '[!] - Please, make sure that you have the weather features available for the specific location!')

        weather_df['date'] = pd.to_datetime(weather_df['date_time'])
        del weather_df['moonrise'], weather_df['moonset'], weather_df['sunrise'], weather_df['sunset']

        weather = []
        for _, row in tqdm(weather_df.iterrows(), total=weather_df.shape[0]):
            values = {
                'date': datetime.strftime(row.date, '%Y-%m-%d'),
                'hour': datetime.strftime(row.date, '%H:%M')
            }
            for columns in weather_df.columns[1:-1]:
                values[columns] = row[columns]
            weather.append(values)

        weather_df = pd.DataFrame(weather)
        return weather_df

    @staticmethod
    def preprocess_readings(data_path: str = 'data/pecanstreet/'):

        data = pd.read_csv(data_path)
        data = data.sort_values(by='local_15min').reset_index(drop=True)
        cid = data['dataid'].unique()[0]
        log(INFO, f"Preprocessing readings from {cid}")

        new_data = data.copy()
        new_data['crop_date'] = pd.to_datetime(new_data['local_15min'], utc=True)
        new_data['generation_solar1'] = np.where(new_data['solar'] < 0, 0, new_data['solar'])
        new_data['generation_solar2'] = np.where(new_data['solar2'] < 0, 0, new_data['solar2'])

        del new_data['solar'], new_data['solar2'], new_data['leg1v'], new_data['leg2v']
        data_columns = list(new_data.columns)

        consumption = data_columns[2:len(data_columns) - 3]
        new_data["sum_consumption"] = new_data[consumption].sum(axis=1)

        generation = data_columns[len(data_columns) - 2:]
        new_data["sum_generation"] = new_data[generation].sum(axis=1)

        compiled = pd.DataFrame(
            {'cid': new_data['dataid'], 'date': new_data['local_15min'], 'consumption': new_data['sum_consumption'],
             'generation': new_data['sum_generation'], 'crop_date': new_data['crop_date']})
        df = compiled.copy()
        df['prev_consumption'] = df.shift(1)['consumption']
        df['consumption_change'] = df.apply(
            lambda row: 0 if np.isnan(row.prev_consumption) else row.consumption - row.prev_consumption, axis=1
        )
        rows = []

        for _, row in df.iterrows():
            date_format = pd.Timestamp(row.date)
            row_data = dict(
                cid=row.cid,
                date=datetime.strftime(row.crop_date, '%Y-%m-%d'),
                hour=datetime.strftime(row.crop_date, '%H:%M'),
                generation=row.generation,
                time_hour=date_format.hour,
                time_minute=date_format.minute,
                month=date_format.month,
                day_of_week=date_format.dayofweek,
                day=date_format.day,
                week_of_year=date_format.week,
                consumption_change=row.consumption_change,
                consumption=row.consumption,
            )
            rows.append(row_data)
        features_df = pd.DataFrame(rows)
        log(INFO, f"{cid} data loaded with shape: {features_df.shape}")
        return features_df, cid


    @classmethod
    def aggregate_features(cls, _id: int, path: str = 'data/pecanstreet/'):
        weather_df = cls.init_weather_readings(path)

        features_df, _id = cls.preprocess_readings(data_path=f"{path}/15min/{str(_id)}.str", weather_df=weather_df)
        mkdir_if_not_exists(f"{path}/aggregate/")
        mkdir_if_not_exists(f"{path}/aggregate/15min")
        del features_df['date'], features_df['hour']
        features_df.to_csv(f"{path}/aggregate/15min/{_id}.csv",index=False)
        return {"_id": _id, "data": features_df}


