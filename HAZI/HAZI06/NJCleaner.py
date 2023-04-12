import pandas as pd
#import os

class NJCleaner:

    def __init__ (self, csv_path:str) -> None:
        self.data = pd.read_csv(csv_path)

    def order_by_scheduled_time(self) -> pd.DataFrame: 
        copied_df = self.data.copy()
        sorted_df = copied_df.sort_values('scheduled_time')

        return sorted_df

    def drop_columns_and_nan(self) -> pd.DataFrame: 
        filtered_df = self.data.copy()
        filtered_df = filtered_df.drop(['from', 'to'], axis=1)
        filtered_df = filtered_df.dropna()

        return filtered_df
    
    def convert_date_to_day(self) -> pd.DataFrame: 
        withdays_df = self.data.copy()
        withdays_df['date'] = pd.to_datetime(withdays_df['date'])
        withdays_df['day'] = withdays_df['date'].dt.day_name()
        withdays_df = withdays_df.drop('date', axis=1)

        return withdays_df
    
    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame: 
        withpartofday_df = self.data.copy()
        withpartofday_df["hour"] = pd.to_datetime(withpartofday_df["scheduled_time"]).dt.hour
        withpartofday_df["part_of_the_day"] = withpartofday_df["hour"].apply(lambda x: 'late_night' if x < 4 else 'early_morning' if x < 8 else 'morning' if x < 12 else 'afternoon' if x < 16 else 'evening' if x < 20 else 'night')
        withpartofday_df = withpartofday_df.drop('hour', axis=1)

        return withpartofday_df
    
    def convert_delay(self) -> pd.DataFrame: 
        withdelay_df = self.data.copy()
        withdelay_df["delay"] = withdelay_df["delay_minutes"].apply(lambda x: '0' if x >= 0 and x < 5 else '1' if x >= 5 else '-1')
 
        return withdelay_df
    
    def drop_unnecessary_columns(self) -> pd.DataFrame: 
        clean_df = self.data.copy()
        clean_df = clean_df.drop(['train_id', 'scheduled_time', 'actual_time', 'delay_minutes'], axis=1)
    
        return clean_df
    
    def save_first_60k(self, path: str) -> pd.DataFrame: 
        copy_df = self.data.copy()
        first60k_df = copy_df.iloc[:60000].copy()
        first60k_df.to_csv(path, index=False)

        return first60k_df

    def prep_df(self, path: str = "data/NJ.csv"):
        self.data = self.order_by_scheduled_time()
        self.data = self.drop_columns_and_nan()
        self.data = self.convert_date_to_day()
        self.data = self.convert_scheduled_time_to_part_of_the_day()
        self.data = self.convert_delay()
        self.data = self.drop_unnecessary_columns()
        self.data = self.save_first_60k(path)

#print(os.getcwd())
#print(os.getcwd()+"\\BEVADAT2022232\\HAZI\HAZI06\\NJ Transit + Amtrak.csv")
#myNJCleaner = NJCleaner(os.getcwd()+"\\BEVADAT2022232\\HAZI\HAZI06\\NJ Transit + Amtrak.csv")
#myNJCleaner.prep_df()

