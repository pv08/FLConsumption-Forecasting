import pandas as pd
import numpy as np
import glob
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from src.utils.functions import mkdir_if_not_exists, create_sequences
from sklearn.preprocessing import MinMaxScaler




class ParticipantData:
    def __init__(self, path: str, _id: int):
        self.path = path
        if not Path(f'{self.path}/pecanstreet/aggregate/').exists():
            self.readings = self.aggregate_features(path=f'{self.path}/pecanstreet/', _id=_id)
        else:
            self.readings = self.catch_data(path=f'{self.path}/pecanstreet/aggregate/15min', _id=_id)


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
    def preprocess_readings(weather_df: pd.DataFrame, data_path: str = 'data/pecanstreet/'):
        def insert_weather_data(date, hour):
            values = {}
            loc = weather_df.loc[(weather_df['date'] == str(date)) & (weather_df['hour'] == f'{str(hour)}:00')]
            for _, row in loc.iterrows():
                for columns in loc.columns[2:-1]:
                    values[columns] = row[columns]
            return values

        data = pd.read_csv(data_path)
        data = data.sort_values(by='local_15min').reset_index(drop=True)
        cid = data['dataid'].unique()[0]
        print(f'[*] - Preprocessing readings from {cid}')
        new_data = data.copy()
        new_data['crop_date'] = pd.to_datetime(new_data['local_15min'], utc=True)
        new_data['generation_solar1'] = np.where(new_data['solar'] < 0, 0, new_data['solar'])
        new_data['generation_solar2'] = np.where(new_data['solar2'] < 0, 0, new_data['solar2'])

        del new_data['dataid'], new_data['solar'], new_data['solar2'], new_data['leg1v'], new_data['leg2v']
        data_columns = list(new_data.columns)

        consumption = data_columns[1:len(data_columns) - 3]
        new_data["sum_consumption"] = new_data[consumption].sum(axis=1)

        generation = data_columns[len(data_columns) - 2:]
        new_data["sum_generation"] = new_data[generation].sum(axis=1)

        compiled = pd.DataFrame(
            {'date': new_data['local_15min'], 'consumption': new_data['sum_consumption'],
             'generation': new_data['sum_generation'], 'crop_date': new_data['crop_date']})
        df = compiled.copy()
        df['prev_consumption'] = df.shift(1)['consumption']
        df['consumption_change'] = df.apply(
            lambda row: 0 if np.isnan(row.prev_consumption) else row.consumption - row.prev_consumption, axis=1
        )
        rows = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            date_format = pd.Timestamp(row.date)
            row_data = dict(
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
            weather_data = insert_weather_data(datetime.strftime(row.crop_date, '%Y-%m-%d'),
                                                    datetime.strftime(row.crop_date, '%H'))
            row_data.update(weather_data)
            rows.append(row_data)
        features_df = pd.DataFrame(rows)

        # date_time = pd.to_datetime(df.pop('date'), format='%Y-%m--%d %H:%M:%S', utc=True)

        # timestamp_s = date_time.map(datetime.timestamp)
        # day = 24 * (60 ** 2)
        # year = (365.2425) * day
        #
        # features_df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        # features_df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        #
        # features_df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        # features_df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        print(f"[+] - {cid} data loaded", f"shape: {features_df.shape}")
        print("[!] - Exporting trainable dataframe")
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

class Participant(ParticipantData):
    def __init__(self, path: str, _id: int, sequence_length: int = 60):
        super(Participant, self).__init__(path=path, _id=_id)
        self.path = path
        self.sequence_length = sequence_length
        self.readings = self.preprocess(readings=self.readings, sequence_length=sequence_length)

    @staticmethod
    def preprocess(readings, sequence_length: int = 60):
        df = readings['data']
        original_data = df.copy()
        n = len(df)

        n_features = len(original_data.columns.to_list())

        train_df = df[0: int(n * .7)]
        val_df = df[int(n * .7): int(n * (1.1 - .2))]
        test_df = df[int(n * (1.0 - .1)):]

        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler = scaler.fit(df)


        train_df = pd.DataFrame(
            scaler.transform(train_df),
            index=train_df.index,
            columns=train_df.columns
        )

        val_df = pd.DataFrame(
            scaler.transform(val_df),
            index=val_df.index,
            columns=val_df.columns
        )

        test_df = pd.DataFrame(
            scaler.transform(test_df),
            index=test_df.index,
            columns=test_df.columns
        )

        readings['train_sequence'] = create_sequences(train_df, 'consumption', sequence_length)
        readings['val_sequence'] = create_sequences(val_df, 'consumption', sequence_length)
        readings['test_sequence'] = create_sequences(test_df, 'consumption', sequence_length)
        readings['scaler'] = scaler
        readings['n_features'] = n_features
        return readings
