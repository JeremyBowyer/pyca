import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats import pearsonr
import time
import psutil
import csv
import datetime
import gc

from models import NumpyModel, PandasModel
import transformations
import filters

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import *

from IPython import embed;

class Worker(QObject):
    
    def __init__(self):
        super().__init__()
        self.__abort = False
    
    def abort(self):
        self.__abort = True


class DataWorker(Worker):
    
    df_loaded_signal = pyqtSignal()
    update_cols_signal = pyqtSignal(list, list)
    analysis_complete_signal = pyqtSignal(object,object,object,object)
    correlation_scatter_complete_signal = pyqtSignal(pd.DataFrame)
    correlation_histogram_complete_signal = pyqtSignal(list)
    loading_msg_signal = pyqtSignal(str)
    start_task_signal = pyqtSignal(str)
    start_status_task_signal = pyqtSignal(str)
    finish_status_task_signal = pyqtSignal()
    show_error_signal = pyqtSignal(str)
    run_metric_dive_signal = pyqtSignal(pd.DataFrame, dict)
    df_model_signal = pyqtSignal(object)
    filter_applied_signal = pyqtSignal(filters.Filter)
    transformation_completed_signal = pyqtSignal(transformations.Transformation)

    def __init__(self):
        super().__init__()
        self.df              = None
        self.data_df          = None
        self.date_cor_df     = None
        self.date_dp_df      = None
        self.all_dates_df    = None
        self.cor_mat_df      = None
        self.metric_df       = None

        self.ByDate          = False
        self.ByCategory      = False

        self.all_dates_model = None
        self.date_cor_model  = None
        self.date_dp_model   = None
        self.cor_mat_model   = None

        self.yCol            = ""
        self.dateCol         = ""
        self.categoryCol     = ""
        self.ignoreCols      = []
        self.multiCols       = []
        self.xCols           = []
        
        self.filters         = {}
        self.transformations = []

    @pyqtSlot(pd.DataFrame)
    def load_df(self, df):
        self.df = df
        self.df_loaded_signal.emit()
        self.update_cols_signal.emit(list(self.df.columns), list(self.xCols))

    @pyqtSlot(dict)
    def run_analysis(self, cols):
        if not self.df_is_loaded():
            return

        if not self.validate_columns(cols):
            return

        a = datetime.datetime.now()

        self.data_df = self.generate_data_df()

        self.process_columns(cols)
        self.run_all_dates()
        self.run_by_date()
        self.create_models()

        b = datetime.datetime.now()
        print(b-a)

        self.analysis_complete_signal.emit(self.all_dates_model, self.date_cor_model, self.date_dp_model, self.cor_mat_model)
        self.update_cols_signal.emit(list(self.df.columns), list(self.xCols))

        self.clear_variables()

    def validate_columns(self, cols):
        if cols["yCol"] == "" or cols["yCol"] not in self.df.columns:
            self.show_error_signal.emit(f"Y Column '{cols['yCol']}' not found in dataset. Please provide a valid column name.")
            return False
        if cols["dateCol"] != "" and cols["dateCol"] not in self.df.columns:
            self.show_error_signal.emit(f"Date Column '{cols['dateCol']}' not found in dataset. Please provide a valid column name.")
            return False
        if cols["categoryCol"] != "" and cols["categoryCol"] not in self.df.columns:
            self.show_error_signal.emit(f"Category Column '{cols['categoryCol']}' not found in dataset. Please provide a valid column name.")
            return False

        for col in cols["ignoreCols"]:
            if col != "" and col not in self.df.columns:
                self.show_error_signal.emit(f"Ignore Column '{col}' not found in dataset. Please provide a valid column name.")
                return False
        
        for col in cols["multiCols"]:
            if col != "" and col not in self.df.columns:
                self.show_error_signal.emit(f"Multi-linear Column '{col}' not found in dataset. Please provide a valid column name.")
                return False

        self.start_task_signal.emit("Calculating summary statistics for metrics...")
        return True

    def process_columns(self, cols):
        self.loading_msg_signal.emit("Initializing analysis...")
        # Store columns
        self.yCol        = cols["yCol"]
        self.dateCol     = cols["dateCol"]
        self.categoryCol = cols["categoryCol"]
        self.ignoreCols  = [self.yCol, self.dateCol, self.categoryCol] + cols["ignoreCols"]
        self.multiCols   = cols["multiCols"]
        self.xCols       = [col for col in self.data_df.columns if col not in self.ignoreCols]

        # coerce to date
        if self.dateCol in self.data_df.columns:
            try:
                self.data_df[self.dateCol] = pd.to_datetime(self.data_df[self.dateCol])
                self.ByDate = True
            except ValueError:
                pass

        if self.categoryCol != "" and self.categoryCol is not None and self.categoryCol in self.data_df.columns:
            self.ByCategory = True

        # coerce to numeric
        for metric in self.xCols + [self.yCol]:
            self.data_df[metric] = pd.to_numeric(self.data_df[metric], errors="coerce")

    def run_all_dates(self):
        metrics     = []
        correls     = []
        rsqrds      = []
        slopes      = []
        pvalues     = []
        counts      = []
        perf_diffs  = []
        turnovers   = []

        cnt = 0
        for metric in self.xCols:
            cnt += 1
            self.loading_msg_signal.emit("Analyzing Metric: " + str(cnt) + " / " + str(len(self.xCols)))
            metric_df = pd.DataFrame({"x":self.data_df[metric],"y":self.data_df[self.yCol]})
            metric_df.dropna(inplace=True)
            
            count = metric_df.shape[0]
            
            if count <= 10:
                continue

            x = metric_df["x"]
            y = metric_df["y"]

            slope, intercept, r_value, p_value, std_err = linregress(x,y)
            if r_value != 0:
                metrics.append(metric)
                correls.append(r_value)
                rsqrds.append(r_value*r_value)
                slopes.append(slope)
                pvalues.append(p_value)
                counts.append(count)
            
                # Performance Differential
                if not self.ByDate:
                    perf_df = self.calculate_performance(x, y)
                else:
                    perf_df = self.calculate_performance(self.data_df[metric], self.data_df[self.yCol], self.data_df[self.dateCol], False)
                perf_diff = perf_df.loc[perf_df['Quintile'] == 'Differential']['All']
                perf_diffs.append(perf_diff)

                # Metric Turnover
                if self.ByDate and self.ByCategory:
                    turnover = self.calculate_metric_turnover(self.data_df[metric], self.data_df[self.dateCol], self.data_df[self.categoryCol])
                else:
                    turnover = np.nan
                turnovers.append(turnover)
        
        cols = ["Metric","Correlation","R-Squared","Slope","P-Value","Data Points", "Performance Differential", "Metric Turnover"]
        self.all_dates_df = pd.DataFrame(columns = cols)

        self.all_dates_df["Metric"]                   = metrics
        self.all_dates_df["Correlation"]              = ['%.4f' % elem for elem in correls]
        self.all_dates_df["R-Squared"]                = ['%.4f' % elem for elem in rsqrds]
        self.all_dates_df["Slope"]                    = ['%.4f' % elem for elem in slopes]
        self.all_dates_df["P-Value"]                  = ['%.4f' % elem for elem in pvalues]
        self.all_dates_df["Data Points"]              = counts
        self.all_dates_df["Performance Differential"] = ['%.4f' % elem for elem in perf_diffs]
        self.all_dates_df["Metric Turnover"]          = ['%.4f' % elem for elem in turnovers]

        self.correlation_scatter_complete_signal.emit(self.all_dates_df[["Metric", "Correlation", "Data Points"]])
        self.correlation_histogram_complete_signal.emit(self.all_dates_df["Correlation"].astype('float64').tolist())

        if len(self.multiCols) > 0:
            self.cor_mat_df = self.data_df[self.multiCols].corr().reset_index().rename(columns={'index':'Metric'})
            self.cor_mat_df = self.cor_mat_df[["Metric"]+list(self.cor_mat_df.Metric)]

    def run_by_date(self):
        if not self.ByDate:
            return
        
        self.loading_msg_signal.emit("Correlations by date...")
        self.create_by_date_dfs()

        self.data_df.groupby(self.dateCol, as_index=False).apply(self.analyze_date).reset_index()
        self.loading_msg_signal.emit("Calculating 'by date' statistics...")
        self.date_cor_df = self.date_cor_df.apply(self.calculate_by_date_stats, axis=1)

    def analyze_date(self, df):
        date_string = datetime.datetime.strftime(list(df[self.dateCol])[0], "%Y-%m-%d")

        self.loading_msg_signal.emit("Analyzing Date: " + date_string)

        cors = []
        dps  = []
        for metric in self.xCols:
            # metric_df = pd.DataFrame({"x": df[metric], "y": df[self.yCol]})
            # metric_df.dropna(inplace=True)

            # count = metric_df.shape[0]


            x = list(df[metric])
            y = list(df[self.yCol])

            no_na_x = []
            no_na_y = []
            for i,val in enumerate(x):
                if not np.isnan(x[i]) and not np.isnan(y[i]):
                    no_na_x.append(x[i])
                    no_na_y.append(y[i])

            count = len(no_na_x)

            if count < 10:
                cors.append(np.nan)
                dps.append(np.nan)
                continue

            r_value, _ = pearsonr(no_na_x, no_na_y)
            cors.append(r_value)
            dps.append(count)

        self.date_cor_df[date_string] = cors
        self.date_dp_df[date_string]  = dps

    def calculate_by_date_stats(self, row):
        periods, neg_periods, pos_periods, perc_neg, perc_pos, wgtd_avg_cor, avg_cor = (0,0,0,0,0,0,0)
        cor_list = []
        for el in row[8:]:
            try:
                cor_list.append(float(el))
            except ValueError:
                cor_list.append(np.nan)

        dp_list = self.date_dp_df.loc[self.date_dp_df["Metric"] == row["Metric"]].values.flatten().tolist()[1:]

        cor_arr = np.array(cor_list) * np.array(dp_list)

        periods = sum([1 for el in cor_list if not np.isnan(el)])
        
        if periods == 0:
            row["Total Periods"] = periods
            return row

        neg_periods  = sum([1 for el in cor_list if el < 0])
        pos_periods  = sum([1 for el in cor_list if el > 0])
        perc_neg     = neg_periods / periods
        perc_pos     = pos_periods / periods
        wgtd_avg_cor = np.nansum(cor_arr) / np.nansum(dp_list)
        avg_cor      = np.nanmean(cor_list)

        row["Total Periods"]            = periods
        row["Negative Periods"]         = neg_periods
        row["Positive Periods"]         = pos_periods
        row["% Negative"]               = perc_neg
        row["% Positive"]               = perc_pos
        row["Weighted Avg Correlation"] = wgtd_avg_cor
        row["Avg Correlation"]          = avg_cor

        return row

    def create_by_date_dfs(self):
        columns = ["Metric","Total Periods","Negative Periods","Positive Periods","% Negative","% Positive","Weighted Avg Correlation","Avg Correlation"]
        dates = sorted(list(set(self.df[self.dateCol])))
        dates_str = [datetime.datetime.strftime(date, "%Y-%m-%d") for date in dates]
        self.date_cor_df = pd.DataFrame(columns = columns+dates_str)
        self.date_dp_df = pd.DataFrame(columns = ["Metric"]+dates_str)

        self.date_cor_df["Metric"] = self.xCols
        self.date_dp_df["Metric"] = self.xCols

    def create_models(self):
        self.all_dates_model = NumpyModel(self.all_dates_df, format_cols={1:(True, True), 2:(False, True), 3:(True, True), 4:(False, False), 5:(False, True)})

        if isinstance(self.cor_mat_df, pd.DataFrame) and not self.cor_mat_df.empty:
            self.cor_mat_model = NumpyModel(self.cor_mat_df)

        if self.date_cor_df is not None:
            cor_format_cols = {}
            for i in range(7, self.date_cor_df.shape[1]):
                cor_format_cols[i] = (True, True)
            self.date_cor_model = NumpyModel(self.date_cor_df, format_cols=cor_format_cols)

        if self.date_dp_df is not None:
            dp_format_cols = {}
            for i in range(1, self.date_dp_df.shape[1]):
                dp_format_cols[i] = (True, True)
            self.date_dp_model = NumpyModel(self.date_dp_df, format_cols=dp_format_cols)

    def clear_variables(self):
        del self.data_df

        del self.all_dates_df
        del self.date_cor_df
        del self.date_dp_df

        del self.all_dates_model
        del self.date_cor_model
        del self.date_dp_model
        
        gc.collect()

    def df_is_loaded(self):
        return(self.df is not None)

    def calculate_metric_turnover(self, x, date, category, include_df = False):
        df = pd.DataFrame({"x": x, "date": date, "cat": category})
        df = df.dropna()
        df = df.drop_duplicates()
        if include_df:
            return pd.DataFrame(df.pivot(index='cat', columns='date').rank(pct=True).diff(axis=1).std())
        else:
            try:
                return df.pivot(index='cat', columns='date').rank(pct=True).diff(axis=1).std().mean()
            except:
                return np.nan

    def create_quintile(self, x):
        quintiles = np.nanpercentile(x, [20,40,60,80])
        return 5 - np.searchsorted(quintiles, x)

    def create_performance_differential(self, df):
        differentials = ['Differential']
        for col in df.columns[1:]:
            try:
                d = ((df[col][0] * 2 + df[col][1]) / 3) - ((df[col][4] * 2 + df[col][3]) / 3)
            except (IndexError, KeyError) as e:
                d = np.nan
            differentials.append(d)
        df.loc[len(df)] = differentials
        return df
        
    def calculate_performance(self, x, y, date_col=None, return_date_cols=True):
        if date_col is None:
            data_df = pd.DataFrame({'x': x, 'y': y})
            data_df['Quintile'] = self.create_quintile(x)
            rtn_df = data_df.groupby('Quintile').mean()['y'].reset_index().rename(columns={"y": "All"})

        else:
            def add_quints(df):
                df['Quintile'] = self.create_quintile(df['x'])
                return df
            
            data_df = pd.DataFrame({'date': date_col, 'x': x, 'y': y})
            data_df = data_df.groupby('date').apply(add_quints)
            rtn_df = data_df.groupby('Quintile').mean()['y'].reset_index().rename(columns={"y": "All"})
            
            if return_date_cols:
                def add_date_performance(df):
                    date_str = df.name.strftime('%Y-%m-%d')
                    rtn_df[date_str] = df.groupby('Quintile').mean().reset_index()['y']
    
                data_df.groupby('date').apply(add_date_performance)
        rtn_df = self.create_performance_differential(rtn_df)
        return rtn_df

    @pyqtSlot(str)
    def run_metric_dive(self, col):
        if not self.df_is_loaded() or self.yCol == '' or col not in self.df.columns:
            return

        cols = {
            "x": col,
            "y": self.yCol,
            "size": self.yCol
            }
            
        self.metric_df = self.df[[col, self.yCol, self.yCol]]
        self.metric_df = self.metric_df.loc[:,~self.metric_df.columns.duplicated()]
        self.metric_df[col] = pd.to_numeric(self.metric_df[col], errors="coerce")
        self.metric_df[self.yCol] = pd.to_numeric(self.metric_df[self.yCol], errors="coerce")
        self.metric_df.dropna(inplace=True)
        self.run_metric_dive_signal.emit(self.metric_df, cols)

    def create_transformation(self, transformation):
        self.start_status_task_signal.emit("Creating " + transformation.name + " transformation.")
        self.df = transformation.create_transformation(self.df)
        self.transformations.append(transformation)
        dfmodel = NumpyModel(self.df)
        self.df_model_signal.emit(dfmodel)
        self.update_cols_signal.emit(list(self.df.columns), list(self.xCols))
        self.finish_status_task_signal.emit()
        self.transformation_completed_signal.emit(transformation)

    def remove_transformation(self, transformation):
        for col in transformation.col_ids:
            if col in self.df.columns:
                self.df.drop(col, axis=1, inplace=True)
        self.transformations.remove(transformation)
        dfmodel = NumpyModel(self.df)
        self.df_model_signal.emit(dfmodel)
        self.update_cols_signal.emit(list(self.df.columns), list(self.xCols))
    
    def apply_filter(self, _filter):
        self.filters[_filter] = _filter.apply_filter(self.df)
        dfmodel = NumpyModel(self.df[self.compile_filters()])
        self.df_model_signal.emit(dfmodel)
        self.filter_applied_signal.emit(_filter)

    def remove_filter(self, _filter):
        self.filters.pop(_filter, None)
        dfmodel = NumpyModel(self.df[self.compile_filters()])
        self.df_model_signal.emit(dfmodel)

    def compile_filters(self):
        keys = list(self.filters.keys())
        if len(keys) == 0:
            return [True] * self.df.shape[0]
        bools = self.filters[keys[0]]
        for key in keys:
            bools = bools & self.filters[key]
        return bools

    def generate_data_df(self):
        return self.df[self.compile_filters()]
    
class LoadCsvWorker(Worker):
    
    df_signal = pyqtSignal(pd.DataFrame)
    df_model_signal = pyqtSignal(object)
    loading_msg_signal = pyqtSignal(str)
    
    def __init__(self, parent, fileName):
        super().__init__()
        self.parent = parent
        self.fileName = fileName

    @pyqtSlot()
    def work(self):
        data = []
        row_limit = 50000
        starting_mem = psutil.virtual_memory().percent
        available_mem = 100-starting_mem
        with open(self.fileName) as f:
            reader = csv.reader(f, quotechar='"')
            cnt = 0
            for i,x in enumerate(reader):
                cnt += 1
                data.append(x)
                # Every 1,000 rows, check memory
                if i%1000 == 0:
                    mem_usage = psutil.virtual_memory().percent
                    self.loading_msg_signal.emit("Rows Loaded: " + str(i) + "   Memory Usage: " + str(mem_usage) + "%")
                    delta_mem = mem_usage - starting_mem
                    # if memory usage is over 95%, stop process
                    if mem_usage > 95:
                        break
                    # if the current process has used over 60%
                    # of available memory, add current data to
                    # dataframe (or create dataframe) and continue
                    if delta_mem >= available_mem*0.6:
                        if cnt == len(data):
                            cols = data.pop(0)
                            df = pd.DataFrame(data, columns=cols)
                        else:
                            temp = pd.DataFrame(data, columns=cols)
                            df = pd.concat([df,temp])
                        # Reset variables for next iteration
                        data = []
                        temp = None
                        starting_mem = psutil.virtual_memory().percent
                        available_mem = 100-starting_mem
         
        # If previous loop never used enough memory
        # to warrant offloading the data to the 
        # dataframe, do so now.
        if len(data) > 0:
            if cnt == len(data):
                cols = data.pop(0)
                df = pd.DataFrame(data, columns=cols)
            else:
                temp = pd.DataFrame(data, columns=cols)
                df = pd.concat([df,temp])
            data = []
            temp = None
         
        self.df_signal.emit(df)
        dfmodel = NumpyModel(df)
        self.df_model_signal.emit(dfmodel)
        