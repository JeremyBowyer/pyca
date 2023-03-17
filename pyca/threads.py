import numpy as np
import pandas as pd
import datetime as dt
from pandas.api.types import is_numeric_dtype
import cProfile
from scipy.stats import linregress, pearsonr, norm
import statsmodels.formula.api as sm
import time
import psutil
import csv
import datetime
import gc
import math 
import IPython

from util import round_default, try_float, is_binary, is_number, create_quintile
from decorators import pyca_profile, handle_error
from models import NumpyModel, PandasModel
from data_objects import NumpyGrouper

import transformations
import filters

from bokeh.plotting import figure, output_file, show, save
from bokeh.models import Panel, ColumnDataSource, DateFormatter
from bokeh.models.widgets import Tabs, DataTable, TableColumn
from bokeh.models.widgets.markups import Div
from bokeh.layouts import column, row
from bokeh.io import curdoc

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, QMessageBox, QApplication

import debugpy
debugpy.debug_this_thread()

class Worker(QObject):
    
    def __init__(self):
        super().__init__()
        self.__abort = False
    
    def abort(self):
        self.__abort = True


class DataWorker(Worker):
    """A class used to perform intensive data analysis and calculations."""

    MIN_CORREL_PERIODS = 10

    data_loaded_signal = pyqtSignal()
    update_cols_signal = pyqtSignal(list, list)
    analysis_complete_signal = pyqtSignal()
    columns_validated_signal = pyqtSignal(bool, list)
    output_models_completed_signal = pyqtSignal(object,object,object,object,object, object)
    correlation_matrix_completed_signal = pyqtSignal(object)
    correlation_scatter_complete_signal = pyqtSignal(pd.DataFrame)
    correlation_histogram_complete_signal = pyqtSignal(list)
    y_col_change_signal = pyqtSignal(str)
    filters_change_signal = pyqtSignal(list)
    loading_msg_signal = pyqtSignal(str)
    start_task_signal = pyqtSignal(str)
    start_status_task_signal = pyqtSignal(str)
    finish_status_task_signal = pyqtSignal()
    show_error_signal = pyqtSignal(str)
    metric_dive_col_change_signal = pyqtSignal(str)
    metric_dive_df_signal = pyqtSignal(pd.DataFrame, dict)
    metric_dive_3d_signal = pyqtSignal(pd.DataFrame, dict)
    metric_dive_anova_stats = pyqtSignal(dict)
    metric_dive_summary_stats = pyqtSignal(dict)
    metric_dive_sens_spec = pyqtSignal(object, object)
    metric_dive_box_and_whisker = pyqtSignal(list)
    metric_dive_dp_by_date = pyqtSignal(object)
    metric_dive_qq_norm = pyqtSignal(object)
    metric_dive_qq_y = pyqtSignal(object)
    metric_dive_metric_turnover = pyqtSignal(object)
    metric_dive_dp_by_category = pyqtSignal(object)
    metric_dive_perf_date_table_signal = pyqtSignal(object)
    metric_dive_perf_category_table_signal = pyqtSignal(object)
    metric_dive_report_signal = pyqtSignal(pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, str)
    metric_dive_started_signal = pyqtSignal()
    metric_dive_complete_signal = pyqtSignal()
    metric_dive_3d_standalone_complete_signal = pyqtSignal()
    date_aggregation_complete_signal = pyqtSignal()
    remove_date_aggregation_complete_signal = pyqtSignal()
    data_preview_model_signal = pyqtSignal(object)
    metric_dive_data_preview_model_signal = pyqtSignal(object)
    filter_applied_signal = pyqtSignal(filters.Filter)
    transformation_completed_signal = pyqtSignal(transformations.Transformation)
    transformation_failed_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.base_df                  = None
        self.date_agg_df              = None
        self.pivot_df                 = None
        self.analyzed_df              = None
        self.date_cor_df              = None
        self.date_dp_df               = None
        self.category_cor_df          = None
        self.category_dp_df           = None
        self.all_dates_df             = None
        self.cor_mat_df               = None
        self.splash_df                = None

        self.by_date                  = False
        self.by_category              = False

        self.all_dates_model          = None
        self.date_cor_model           = None
        self.category_cor_model       = None
        self.date_dp_model            = None
        self.category_dp_model        = None
        self.cor_mat_model            = None
        self.splash_model             = None

        self.date_col                 = ""
        self.category_col             = ""
        self.ignore_cols              = []
        self.multi_cols               = []
        self.cor_mat_cols             = []

        self._metric_dive_cols        = {"x_col": "", "y_col": "", "date_col": "", "category_col": "", "z_col": ""}

        self.filters                  = {}
        self.metric_dive_filters      = {}
        self.transformations          = []

        self._y_col                   = ""
        
        self.metric_dive_anova_dict = None

    def show_error(self, msg=""):
        """Wrapper method for emitting signal containing a message describing an encountered problem."""

        self.show_error_signal.emit(msg)

    def finish_status_task(self, *args, **kwargs):
        """Wrapper method for emitting signal indicating the current task has been completed."""

        self.finish_status_task_signal.emit()

    def clean_df(self, df, by_date=False, date_col=None, by_category=False, category_col=None):
        """Method used to convert columns to proper data types
        
        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame to be transformed

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame of the same shape and with the same columns,
            but with appropriate columns converted
        """

        df = df.loc[:,~df.columns.duplicated()]
        cols_dict = df.to_dict('series')
        # coerce to date
        if by_date and isinstance(date_col, str):
            try:
                cols_dict[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            except ValueError:
                pass
        
        if by_category and isinstance(category_col, str):
            cols_dict[category_col] = df[category_col]

        # coerce to numeric
        for metric in self.x_cols + [self.y_col]:
            try:
                col = df[metric]
            except KeyError:
                print("Warning! Column: '" + metric + "' not found in main dataframe.")
                continue

            if is_numeric_dtype(col):
                cols_dict[metric] = col
                continue
            try:
                cols_dict[metric] = pd.to_numeric(col, errors="coerce")
            except Exception as e:
                print('Error while converting ' + metric + ' to numeric. /n' + str(e))

        return pd.DataFrame(cols_dict)

    def get_current_df(self, apply_filter=True, clean=False):
        """Gets the appropriate dataframe, given the current state of transformation

        Returns a dataframe given the current state of transformation. 
        If data is aggregated by date, return the date_agg_df dataframe,
        if data is pivoted, return the pivot_df dataframe, 
        otherwise return the base_df dataframe.
        
        Parameters
        ----------
        apply_filter : bool, optional
            A flag used to determine if the dataframe should be filtered, 
            if there are existing filters (default is True)

        clean : bool, optional
            A flag used to determine if the dataframe should be cleaned
            (default is False)

        Returns
        -------
        pd.DataFrame
        """

        if self.is_date_aggregated:
            df = self.date_agg_df
        else:
            if self.is_pivoted:
                df = self.pivot_df
            else:
                df = self.base_df

        if df is None:
            return pd.DataFrame()

        if apply_filter:
            df = df[self.compile_filters(df)]

        if clean:
            df = self.clean_df(df, self.by_date, self.date_col, self.by_category, self.category_col)

        return df.reset_index(drop=True)

    def set_current_df(self, df):
        """Set the current dataframe, determined by the current state of transformation
        
        Sets a dataframe given the current state of transformation. 
        If data is aggregated by date, set the date_agg_df dataframe,
        if data is pivoted, set the pivot_df dataframe, 
        otherwise set the base_df dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame that is to be used as new dataframe going forward
        """

        if self.is_date_aggregated:
            self.date_agg_df = df
        else:
            if self.is_pivoted:
                self.pivot_df = df
            else:
                self.base_df = df

    @property
    def metric_dive_by_date(self):
        base_df = self.get_current_df(apply_filter=True, clean=False)
        if not isinstance(self.metric_dive_date_col, str):
            return False
        
        if not self.metric_dive_date_col in base_df.columns:
            return False
        
        if not pd.to_datetime(base_df[self.metric_dive_date_col], errors="coerce").dropna().shape[0] > 0:
            return False

        return True

    @property
    def metric_dive_by_category(self):
        base_df = self.get_current_df(apply_filter=True, clean=False)

        if not isinstance(self.metric_dive_category_col, str):
            return False

        if not self.metric_dive_category_col in base_df.columns:
            return False

        if not base_df[self.metric_dive_category_col].dropna().shape[0] > 0:
            return False

        return True

    @property
    def metric_dive_cols(self):
        """Return the private variable dict of current metric dive columns"""
        return self._metric_dive_cols

    @metric_dive_cols.setter
    def metric_dive_cols(self, value):
        """Incorporate given dict into metric_dive_cols dict"""
        if not isinstance(value, dict):
            print("Trying to set metric dive cols with non-dict variable")
            return

        for key in value:
            if key in self._metric_dive_cols:
                self._metric_dive_cols[key] = value[key]
            else:
                print("Trying to pass a dict with invalid key name to metric dive cols")

    @property
    def metric_dive_x_col(self):
        """Return a string of the current metric dive X column. Return empty string if no
        current metric dive column exists"""
        return self.metric_dive_cols.get("x_col", "")

    @metric_dive_x_col.setter
    def metric_dive_x_col(self, value):
        """Set the current 'x' column for metric dive
        
        If value is valid (must be a str and different from the current value),
        set 'x' column to be value.

        Parameters
        ----------
        value : str
            Value to be set as 'x' column for metric dive. Must be a str,
            and must not be the existing value.
        """

        if value == self.metric_dive_cols["x_col"] or not isinstance(value, str):
            return
        else:
            self.metric_dive_cols["x_col"] = value

    @property
    def metric_dive_y_col(self):
        """Return a string of the current metric dive Y column. Return empty string if no
        current metric dive column exists"""
        return self.metric_dive_cols.get("y_col", "")

    @metric_dive_y_col.setter
    def metric_dive_y_col(self, value):
        """Set the current 'y' column for metric dive
        
        If value is valid (must be a str and different from the current value),
        set 'y' column to be value.

        Parameters
        ----------
        value : str
            Value to be set as 'y' column for metric dive. Must be a str,
            and must not be the existing value.
        """

        if value == self.metric_dive_cols["y_col"] or not isinstance(value, str):
            return
        else:
            self.metric_dive_cols["y_col"] = value

    @property
    def metric_dive_date_col(self):
        """Return a string of the current metric dive Date column. Return empty string if no
        current metric dive column exists"""
        return self.metric_dive_cols.get("date_col", "")

    @metric_dive_date_col.setter
    def metric_dive_date_col(self, value):
        """Set the current 'date' column for metric dive
        
        If value is valid (must be a str and different from the current value),
        set 'date' column to be value.

        Parameters
        ----------
        value : str
            Value to be set as 'date' column for metric dive. Must be a str,
            and must not be the existing value.
        """

        if value == self.metric_dive_cols["date_col"] or not isinstance(value, str):
            return
        else:
            self.metric_dive_cols["date_col"] = value

    @property
    def metric_dive_category_col(self):
        """Return a string of the current metric dive Category column. Return empty string if no
        current metric dive column exists"""
        return self.metric_dive_cols.get("category_col", "")

    @metric_dive_category_col.setter
    def metric_dive_category_col(self, value):
        """Set the current 'date' column for metric dive
        
        If value is valid (must be a str and different from the current value),
        set 'category' column to be value.

        Parameters
        ----------
        value : str
            Value to be set as 'category' column for metric dive. Must be a str,
            and must not be the existing value.
        """

        if value == self.metric_dive_cols["category_col"] or not isinstance(value, str):
            return
        else:
            self.metric_dive_cols["category_col"] = value

    @property
    def x_cols(self):
        """Return a list of strings that are the names of 'x' columns in the dataset."""

        if self.analyzed_df is None:
            return []
        return [col for col in self.analyzed_df.columns if col not in self.ignore_cols]

    def get_metric_df(self, apply_filter=True, drop_na=True, inf_to_na=True, date_as_str=False, include_cols=[], clean=True):
        """Return a dataframe that is a subset of the current dataframe, using only relevant columns."""

        base_df = self.get_current_df(apply_filter=True, clean=False)

        # Determine abort situations
        if self.metric_dive_x_col not in base_df.columns:
            return pd.DataFrame({})

        if self.metric_dive_y_col not in base_df.columns:
            return pd.DataFrame({})

        # Create list of columns
        col_list = list(self.metric_dive_cols.values())

        if isinstance(include_cols, str):
            include_cols = [include_cols]

        includes = [col for col in include_cols if col in base_df.columns]
        col_list += includes
        col_list = [col for col in col_list if col != "" and isinstance(col, str)]


        # Create dataframe
        if any([col not in base_df.columns for col in col_list]):
            return pd.DataFrame(columns=col_list)

        df = base_df[col_list]
        df = df.reset_index(drop=True)

        if df.shape[0] > 0 and apply_filter:
            df = df[self.compile_metric_dive_filters(df)]
        df = df.loc[:,~df.columns.duplicated()]

        # Clean data, if requested
        if clean:
            for col in df.columns:
                if col in [self.metric_dive_date_col, self.metric_dive_category_col]:
                    continue
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert date column, if necessary
        if self.metric_dive_by_date:
            df[self.metric_dive_date_col] = pd.to_datetime(df[self.metric_dive_date_col], errors="coerce")
            # Convert date column to string, if requested
            if date_as_str:
                try:
                    df[self.metric_dive_date_col] = df[self.metric_dive_date_col].dt.strftime('%Y-%m-%d')
                except Exception as e:
                    print(str(e) + " -- On 'get_metric_df()' trying to convert date column " + self.metric_dive_date_col + " to string.")

        # Remove infs, if requested
        if inf_to_na:
            df = df.replace([np.inf, -np.inf], np.nan)

        # Drop NAs, if requested
        if drop_na:
            df.dropna(inplace=True)

        
        return df

    @property
    def y_col(self):
        """Return the current 'y' column."""

        return self._y_col

    @y_col.setter
    def y_col(self, value):
        """Set the current 'y' column
        
        If value is valid (must be a str and different from the current value),
        set 'y' column to be value and emit signal, indicating it has changed.

        Parameters
        ----------
        value : str
            Value to be set as 'y' column. Must be a str,
            and must not be the existing value.
        """

        if value == self._y_col or not isinstance(value, str):
            return
        else:
            self._y_col = value
            self.y_col_change_signal.emit(self.y_col)

    @property
    def is_date_aggregated(self):
        """Return bool indicating if data is currently aggregated by date."""

        return self.date_agg_df is not None

    @property
    def is_pivoted(self):
        """Return bool indicating if data is currently pivoted."""

        return self.pivot_df is not None

    @pyqtSlot(pd.DataFrame, str)
    def load_df(self, df, file_nm):
        """Set current base_df, and call related methods and signals
        
        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame to be set as the new base_df
        """

        self.remove_all_filters()
        self.remove_metric_dive_filters(False)
        self.base_df = df
        self.data_loaded_signal.emit()
        self.update_cols_signal.emit(list(self.base_df.columns), list(self.x_cols))

    @pyca_profile
    @pyqtSlot(dict)
    def run_analysis(self, cols):
        """Run the plethora of analyses to be done on the current dataset
        
        This is the meat of the app, aside from the metric dive calculations. 
        Runs various calculations for each 'x' column compared to the 'y' column, 
        which includes calculations for the full panel data, and also calculations 
        by date (if provided) and by category (if provided).

        Parameters
        ----------
        cols : dict
            A dict containing the column names for each type of column 
            needed to run the various calculations
        """

        if not self.data_is_loaded:
            self.analysis_complete_signal.emit()
            return

        current_df = self.get_current_df(apply_filter=True, clean=False)
        self.start_task_signal.emit("Calculating summary statistics for metrics...")

        self.analyzed_df = None
        current_df = self.process_columns(cols, current_df)
        self.analyzed_df = current_df
        self.analyzed_df = self.clean_df(self.analyzed_df, self.by_date, self.date_col, self.by_category, self.category_col)

        self.run_all_dates(self.analyzed_df)
        self.run_by_date(self.analyzed_df)
        self.run_by_category(self.analyzed_df)
        self.create_models()

        self.loading_msg_signal.emit("Finishing up...")
        self.analysis_complete_signal.emit()
        self.update_cols_signal.emit(list(self.analyzed_df.columns), list(self.x_cols))

        self.clear_variables()

    @pyqtSlot(dict)
    def validate_columns(self, cols, df=None, emit=True):
        """Checks the validity of user-provided columns
        
        Makes sure the columns that the user provided are valid 
        in various ways. They need to be in the dataset (they 
        should be, given the current logic of the app), and for 
        a given date column, checks to see if the values in that 
        column can be converted to date.

        If all columns are valid, emit a signal, indicating such.

        Parameters
        ----------
        cols : dict
            A dict containing the column names for each type of column 
        df : pd.DataFrame, optional
            pd.DataFrame used to check the validity of the provided columns
        emit : bool, optional
            A bool indicating if a signal should be emitted, indicating
            a successful validation

        Returns
        -------
        bool
            A bool indicating the validity of the provided columns
        """

        df = self.get_current_df(apply_filter=True, clean=False)

        current_cols = df.columns
        confirm_msgs = []

        if cols["y_col"] == "" or cols["y_col"] not in current_cols:
            self.show_error(f"Y Column '{cols['y_col']}' not found in dataset. Please provide a valid column name.")
            self.columns_validated_signal.emit(False, confirm_msgs)
            return False

        if cols["date_col"] != "":
            if cols["date_col"] not in current_cols:
                self.show_error(f"Date Column '{cols['date_col']}' not found in dataset. Please provide a valid column name.")
                self.columns_validated_signal.emit(False, confirm_msgs)
                return False
            try:
                pre_cnt  = df[cols["date_col"]].dropna().unique().shape[0]
                post_cnt = pd.to_datetime(df[cols["date_col"]], errors='coerce').dropna().unique().shape[0]
                if post_cnt == 0:
                    self.show_error(f"After transforming column '{cols['date_col']}' into a date format, there were no valid dates found. Please make sure your data is in a proper date format. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior")
                    self.columns_validated_signal.emit(False, confirm_msgs)
                    return False

                if post_cnt < pre_cnt:
                    confirm_msgs.append(f"After transforming column '{cols['date_col']}' into a date format, some values could not be converted successfully, resulting in NAs.")
            except:
                self.show_error(f"There was a problem transforming column '{cols['date_col']}' into a date format. Please check")
                self.columns_validated_signal.emit(False, confirm_msgs)
                return False
                
        if cols["category_col"] != "" and cols["category_col"] not in current_cols:
            self.show_error(f"Category Column '{cols['category_col']}' not found in dataset. Please provide a valid column name.")
            self.columns_validated_signal.emit(False, confirm_msgs)
            return False

        for col in cols["ignore_cols"]:
            if col != "" and col not in current_cols:
                self.show_error(f"Ignore Column '{col}' not found in dataset. Please provide a valid column name.")
                self.columns_validated_signal.emit(False, confirm_msgs)
                return False
        
        for col in cols["multi_cols"]:
            if col != "" and col not in current_cols:
                self.show_error(f"Multi-linear Column '{col}' not found in dataset. Please provide a valid column name.")
                self.columns_validated_signal.emit(False, confirm_msgs)
                return False

        # if cols["category_col"] != "" and cols["category_col"] in current_cols:
        #     cat_count = df[cols["category_col"]].unique().shape[0]
        #     if cat_count > 50:
        #         confirm_msgs.append("Category column has " + str(cat_count) + " unique values. This could cause the app to run slowly at times.")

        # confirm_msgs.append("also another problem")

        if emit:
            self.columns_validated_signal.emit(True, confirm_msgs)
        return True

    def process_columns(self, cols, df):
        """Processes a given column dict
        
        Takes the column information in the user-provided dict, 
        and stores the information in class variables. Also sets
        class variables indicating if the data is to be calculated 
        by date and by category.

        Parameters
        ----------
        cols : dict
            A dict containing the column names for each type of column 
        df : pd.DataFrame
            pd.DataFrame used to process given columns

        Returns
        -------
        pd.DataFrame
            The provided pd.DataFrame but with appropriate columns converted
        """

        self.loading_msg_signal.emit("Initializing analysis...")
        # Store columns
        self.y_col        = cols["y_col"]
        self.date_col     = cols["date_col"]
        self.category_col = cols["category_col"]
        self.ignore_cols  = [self.y_col, self.date_col, self.category_col] + cols["ignore_cols"]
        self.multi_cols   = cols["multi_cols"]

        if self.category_col != "" and self.category_col is not None and self.category_col in df.columns:
            self.by_category = True
        else:
            self.by_category = False

        if self.date_col != "" and self.date_col is not None and self.date_col in df.columns:
            try:
                df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
                self.by_date = True
            except:
                self.by_date = False
        else:
            self.by_date = False

        return df

    def run_all_dates(self, df):
        """Run calculations for the full panel of data
        
        For each 'x' column, run a slew of regression stats for all years and categories.

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame containing the data to be analyzed
        """

        metrics     = []
        correls     = []
        rsqrds      = []
        slopes      = []
        pvalues     = []
        counts      = []
        perf_diffs  = []
        turnovers   = []

        if len(self.multi_cols) > 0:
            df['Multi-Linear'] = self.create_multi_linear_pred(df[self.multi_cols + [self.y_col]], self.y_col)

        # Single Linear Regressions
        cnt = 0
        for metric in self.x_cols:
            cnt += 1
            self.loading_msg_signal.emit("Analyzing Metric: " + str(cnt) + " / " + str(len(self.x_cols)))
            metric_df = pd.DataFrame({"x":df[metric],"y":df[self.y_col]})
            if self.by_date:
                metric_df['Date'] = df[self.date_col]
            metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Check for an acceptable number of data points
            count = metric_df.shape[0]
            if count <= 10:
                metrics.append(metric)
                correls.append(np.nan)
                rsqrds.append(np.nan)
                slopes.append(np.nan)
                pvalues.append(np.nan)
                counts.append(count)
                perf_diffs.append(np.nan)
                turnovers.append(np.nan)
                continue

            x = metric_df["x"]
            y = metric_df["y"]
            slope, intercept, r_value, p_value, std_err = linregress(x,y)

            # Check for valid output
            if r_value == 0:
                metrics.append(metric)
                correls.append(np.nan)
                rsqrds.append(np.nan)
                slopes.append(np.nan)
                pvalues.append(np.nan)
                counts.append(count)
                perf_diffs.append(np.nan)
                turnovers.append(np.nan)
                continue

            # Fill in stats
            metrics.append(metric)
            correls.append(r_value)
            rsqrds.append(r_value*r_value)
            slopes.append(slope)
            pvalues.append(p_value)
            counts.append(count)
        
            # Performance Differential
            if not self.by_date:
                perf_df = self.calculate_performance(x, y)
            else:
                perf_df = self.calculate_performance(metric_df['x'], metric_df['y'], metric_df['Date'], False)

            try:
                perf_diff = perf_df.loc[perf_df[perf_df.columns[0]] == 'Differential']['All']
                perf_diffs.append(perf_diff)
            except:
                perf_diffs.append(np.nan)

            # Metric Turnover
            if self.by_date and self.by_category:
                turnover = self.calculate_metric_turnover(df[metric], df[self.date_col], df[self.category_col])
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
        self.correlation_histogram_complete_signal.emit(self.all_dates_df["Correlation"].astype('float64').replace([np.inf, -np.inf], np.nan).dropna().tolist())

    def create_multi_linear_pred(self, df, y_col):
        """Create a list-like object containing predicted Ys for a set of Xs

        From: https://stackoverflow.com/questions/19991445/run-an-ols-regression-with-pandas-data-frame

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame containing the data to be analyzed. Must not contain
            NAs or Infs

        y_col : str
            A str identifying the Y column, all other columns will be assumed
            to be X columns
        """

        multi_df = df.copy()
        multi_df = multi_df.loc[:,~multi_df.columns.duplicated()]
        for col in multi_df.columns:
            multi_df[col] = pd.to_numeric(multi_df[col], errors='coerce')
        multi_df = multi_df.replace([np.inf, -np.inf], np.nan).dropna()
        count = multi_df.shape[0]
        if count <= 3:
            self.show_error('One or more columns chosen for multilinear regression has insufficient numeric data.')
            return [np.nan] * df.shape[0]
        else:
            col_rename = {y_col: 'y'}
            cnt = 1
            for col in multi_df.columns:
                if col == y_col:
                    continue
                col_rename[col] = 'x'+str(cnt)
                cnt += 1
            multi_df.rename(col_rename, axis=1, inplace=True)

            # Create Model
            form = "y ~ " + " + ".join(list(col_rename.values())[1:])
            result = sm.ols(formula=form, data=multi_df).fit()

            # Extract Coefs
            coef_table = result.summary().tables[1].data
            coefs = {'intercept': float(coef_table[1][1])}
            for i in range(2,len(coef_table)):
                coefs[coef_table[i][0]] = coef_table[i][1]

            s = pd.Series([coefs['intercept']]*df.shape[0])
            for x in list(col_rename.keys())[1:]:
                s = s + df[x] * float(coefs[col_rename[x]])
            return s

    def run_by_category(self, df):
        """Run various stats for each 'x' column in dataset, for each category, if a category column was provided."""

        if not self.by_category:
            return

        self.loading_msg_signal.emit("Correlations by category...")
        QApplication.processEvents()
        self.create_by_category_dfs()

        def analyze_category(grp):
            category_string = str(grp.name)
            self.loading_msg_signal.emit("Analyzing Category: " + category_string)
            QApplication.processEvents()
            
            cors = []
            dps  = []
            for metric in self.x_cols:
                metric_df = grp[[metric, self.y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                count = metric_df.shape[0]
                no_na_x = metric_df[metric]
                no_na_y = metric_df[self.y_col]

                # x = list(grp[metric])
                # y = list(grp[self.y_col])

                # no_na_x = []
                # no_na_y = []
                # for i,val in enumerate(x):
                #     if is_number(x[i]) and is_number(y[i]):
                #         no_na_x.append(x[i])
                #         no_na_y.append(y[i])

                # count = len(no_na_x)

                if count < self.MIN_CORREL_PERIODS:
                    cors.append(np.nan)
                    dps.append(np.nan)
                    continue

                r_value, _ = pearsonr(no_na_x, no_na_y)
                cors.append(r_value)
                dps.append(count)

            self.category_cor_df[category_string] = cors
            self.category_dp_df[category_string]  = dps
            QApplication.processEvents()

        self.loading_msg_signal.emit("Calculating 'by category' statistics...")
        df.groupby(self.category_col, as_index=False).apply(analyze_category).reset_index()

        def calculate_by_category_stats(row):
            periods, neg_periods, pos_periods, perc_neg, perc_pos, wgtd_avg_cor, avg_cor = (0,0,0,0,0,0,0)
            cor_list = []
            for el in row[8:]:
                try:
                    cor_list.append(float(el))
                except ValueError:
                    cor_list.append(np.nan)

            dp_list = self.category_dp_df.loc[self.category_dp_df["Metric"] == row["Metric"]].values.flatten().tolist()[1:]

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
            QApplication.processEvents()
            return row

        self.category_cor_df = self.category_cor_df.apply(calculate_by_category_stats, axis=1)

    def run_by_date(self, df):
        """Run various stats for each 'x' column in dataset, for each date, if a date column was provided."""

        if not self.by_date:
            return
        
        self.loading_msg_signal.emit("Correlations by date...")
        QApplication.processEvents()
        self.create_by_date_dfs()

        def analyze_date(grp):
            date_string = datetime.datetime.strftime(list(grp[self.date_col])[0], "%Y-%m-%d")
            #print('Analyzing Date: ' + date_string)
            self.loading_msg_signal.emit("Analyzing Date: " + date_string)
            QApplication.processEvents()
            
            cors = []
            dps  = []
            for metric in self.x_cols:
                metric_df = grp[[metric, self.y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                count = metric_df.shape[0]
                no_na_x = metric_df[metric]
                no_na_y = metric_df[self.y_col]

                # x = list(grp[metric])
                # y = list(grp[self.y_col])

                # no_na_x = []
                # no_na_y = []
                # for i,val in enumerate(x):
                #     if is_number(x[i]) and is_number(y[i]):
                #         no_na_x.append(x[i])
                #         no_na_y.append(y[i])

                # count = len(no_na_x)

                if count < self.MIN_CORREL_PERIODS:
                    cors.append(np.nan)
                    dps.append(np.nan)
                    continue

                r_value, _ = pearsonr(no_na_x, no_na_y)
                cors.append(r_value)
                dps.append(count)

            self.date_cor_df[date_string] = cors
            self.date_dp_df[date_string]  = dps
            QApplication.processEvents()

        self.loading_msg_signal.emit("Calculating 'by date' statistics...")
        df.groupby(self.date_col, as_index=False).apply(analyze_date).reset_index()

        def calculate_by_date_stats(row):
            periods, neg_periods, pos_periods, perc_neg, perc_pos, wgtd_avg_cor, avg_cor = (0,0,0,0,0,0,0)
            cor_list = []
            for el in row[8:]:
                try:
                    cor_list.append(float(el))
                except ValueError:
                    cor_list.append(np.nan)

            dp_list = self.date_dp_df.loc[self.date_dp_df["Metric"] == row["Metric"]].values.flatten().tolist()[1:]

            # TODO: this whole thing sucks
            cor_list_complete_cases = []
            dp_list_complete_cases  = []
            for pair in zip(cor_list, dp_list):
                if np.isnan(pair[0]) or np.isnan(pair[1]):
                    continue
                cor_list_complete_cases.append(pair[0])
                dp_list_complete_cases.append(pair[1])

            cor_arr_complete_cases = np.array(cor_list_complete_cases) * np.array(dp_list_complete_cases)

            periods = sum([1 for el in cor_list if not np.isnan(el)])
            
            if periods == 0:
                row["Total Periods"] = periods
                return row

            neg_periods  = sum([1 for el in cor_list if el < 0])
            pos_periods  = sum([1 for el in cor_list if el > 0])
            perc_neg     = neg_periods / periods
            perc_pos     = pos_periods / periods
            wgtd_avg_cor = np.nansum(cor_arr_complete_cases) / np.nansum(dp_list_complete_cases)
            avg_cor      = np.nanmean(cor_list)


            row["Total Periods"]            = periods
            row["Negative Periods"]         = neg_periods
            row["Positive Periods"]         = pos_periods
            row["% Negative"]               = perc_neg
            row["% Positive"]               = perc_pos
            row["Weighted Avg Correlation"] = wgtd_avg_cor
            row["Avg Correlation"]          = avg_cor
            QApplication.processEvents()
            return row

        self.date_cor_df = self.date_cor_df.apply(calculate_by_date_stats, axis=1)

    def create_by_category_dfs(self):
        """Creates empty dataframes, to be filled in with the 'run_by_category' method."""
        columns = ["Metric","Total Periods","Negative Periods","Positive Periods","% Negative","% Positive","Weighted Avg Correlation","Avg Correlation"]
        categories = sorted(list(set(self.analyzed_df[self.category_col])))
        categories_str = [str(category) for category in categories]
        self.category_cor_df = pd.DataFrame(columns = columns+categories_str)
        self.category_dp_df = pd.DataFrame(columns = ["Metric"]+categories_str)

        self.category_cor_df["Metric"] = self.x_cols
        self.category_dp_df["Metric"] = self.x_cols

    def create_by_date_dfs(self):
        """Creates empty dataframes, to be filled in with the 'run_by_date' method."""

        columns = ["Metric","Total Periods","Negative Periods","Positive Periods","% Negative","% Positive","Weighted Avg Correlation","Avg Correlation"]
        dates = sorted(list(set(self.analyzed_df[self.date_col])))
        dates_str = [datetime.datetime.strftime(date, "%Y-%m-%d") for date in dates]
        self.date_cor_df = pd.DataFrame(columns = columns+dates_str)
        self.date_dp_df = pd.DataFrame(columns = ["Metric"]+dates_str)

        self.date_cor_df["Metric"] = self.x_cols
        self.date_dp_df["Metric"] = self.x_cols

    def create_models(self):
        """Create models and emit models for the dataframes containing calculated statistics."""

        self.loading_msg_signal.emit("Creating output tables")

        # Splash
        try:
            if self.by_date:
                splash_cols = [
                    "Metric",
                    "Metric Turnover",
                    "Weighted Avg Correlation",
                    "Data Points",
                    "Performance Differential",
                    "Total Periods",
                    "% Negative",
                    "% Positive"
                ]
                self.splash_df = pd.merge(self.all_dates_df, self.date_cor_df, how="left", on="Metric")[splash_cols]
                self.splash_model = NumpyModel(self.splash_df, format_cols={1:(False, False), 2:(True, True), 3:(False, True), 4:(False, True), 5:(False, True), 6:(False, True)})
            else:
                splash_cols = [
                    "Metric",
                    "Data Points",
                    "Performance Differential",
                ]
                self.splash_df = self.all_dates_df[splash_cols]
                self.splash_model = NumpyModel(self.splash_df, format_cols={1:(False, True), 2:(False, True)})
        except Exception as e:
            print("error creating splash page model -- " + str(e))
            self.splash_df = None
            self.splash_model = None

        # All Dates
        self.all_dates_model = NumpyModel(self.all_dates_df, format_cols={1:(True, True), 2:(False, True), 3:(True, True), 4:(False, False), 5:(False, True), 6:(True, True), 7:(False, False)})

        # By Date - correlations
        if hasattr(self, 'date_cor_df') and self.date_cor_df is not None:
            cor_format_cols = {}
            for i in range(6, self.date_cor_df.shape[1]):
                cor_format_cols[i] = (True, True)
            self.date_cor_model = NumpyModel(self.date_cor_df, format_cols=cor_format_cols)
        else:
            self.date_cor_model = None

        # By Category - correlations
        if hasattr(self, 'category_cor_df') and self.category_cor_df is not None:
            cor_format_cols = {}
            for i in range(6, self.category_cor_df.shape[1]):
                cor_format_cols[i] = (True, True)
            self.category_cor_model = NumpyModel(self.category_cor_df, format_cols=cor_format_cols)
        else:
            self.category_cor_model = None
            
        # By Date - data points
        if hasattr(self, 'date_dp_df') and self.date_dp_df is not None:
            dp_format_cols = {}
            for i in range(1, self.date_dp_df.shape[1]):
                dp_format_cols[i] = (True, True)
            self.date_dp_model = NumpyModel(self.date_dp_df, format_cols=dp_format_cols)
        else:
            self.date_dp_model = None

        # By Category - data points
        if hasattr(self, 'category_dp_df') and self.category_dp_df is not None:
            dp_format_cols = {}
            for i in range(1, self.category_dp_df.shape[1]):
                dp_format_cols[i] = (True, True)
            self.category_dp_model = NumpyModel(self.category_dp_df, format_cols=dp_format_cols)
        else:
            self.category_dp_model = None

        self.output_models_completed_signal.emit(self.splash_model, self.all_dates_model, self.date_cor_model, self.category_cor_model, self.date_dp_model, self.category_dp_model)

    def run_correlation_matrix(self, cols):
        self.start_status_task_signal.emit("Creating correlation matrix...")
        df = self.get_current_df(apply_filter=True, clean=False)

        # Validate columns
        for col in cols:
            if col != "" and col not in df.columns:
                self.show_error(f"Correlation Matrix Column '{col}' not found in dataset. Please provide a valid column name.")
                self.correlation_matrix_completed_signal.emit(None)
                self.finish_status_task()
                return False

        # Create DF
        if len(cols) > 0:
            self.cor_mat_df = df[cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).corr().reset_index().rename(columns={'index':'Metric'})
            self.cor_mat_df = self.cor_mat_df[["Metric"]+list(self.cor_mat_df.Metric)]

        # Create Model
        if isinstance(self.cor_mat_df, pd.DataFrame) and not self.cor_mat_df.empty:
            self.cor_mat_model = NumpyModel(self.cor_mat_df)
        else:
            self.cor_mat_model = None

        # Emit Model
        self.correlation_matrix_completed_signal.emit(self.cor_mat_model)
        self.finish_status_task()

    def clear_variables(self):
        """Method for clearing class variables, to save memory."""

        if hasattr(self, 'all_dates_df'):
            del self.all_dates_df
        if hasattr(self, 'date_cor_df'):
            del self.date_cor_df
        if hasattr(self, 'date_dp_df'):
            del self.date_dp_df
        if hasattr(self, 'category_cor_df'):
            del self.category_cor_df
        if hasattr(self, 'category_dp_df'):
            del self.category_dp_df

        if hasattr(self, 'all_dates_model'):
            del self.all_dates_model
        if hasattr(self, 'date_cor_model'):
            del self.date_cor_model
        if hasattr(self, 'category_cor_model'):
            del self.category_cor_model
        if hasattr(self, 'date_dp_model'):
            del self.date_dp_model
        if hasattr(self, 'category_dp_model'):
            del self.category_dp_model
        
        gc.collect()

    @property
    def data_is_loaded(self):
        """Return bool indicating if there is data loaded into the app."""

        return self.base_df is not None

    def calculate_metric_turnover(self, x, date, category, include_df = False):
        """Calculates a custom statistic called 'metric turnover' for a given metric
        
        To calculate 'metric turnover' you need an x, date and category column. 
        An x factor's metric turnover score is the standard deviation of the 
        change in percentile over time, where the percentile is calculated by 
        the given category identifier column for each date.

        This list can be averaged when looking at the metric's score overall, 
        as opposed to looking at it by date.

        Parameters
        ----------
        x : pd.Series
            A list-like iterable (probably a pd.Series) of the x factor
        date : pd.Series
            A list-like iterable (probably a pd.Series) of the date identifier
        category : pd.Series
            A list-like iterable (probably a pd.Series) of the category identifier
        include_df : bool, optional
            A bool indicating of the entire history of the metric turnover score should be included

        Returns
        -------
        if include_df == True
            pd.DataFrame
                A pd.DataFrame with a row for each date, and the accompanying, 
                metric turnover for that date.
        else
            float
                The mean of the metric turnovers described in the pd.DataFrame, 
                described above.
        """

        df = pd.DataFrame({"x": x, "date": date, "cat": category})
        df = df.dropna()
        df = df.drop_duplicates()
        if include_df:
            try:
                rtrn_df = pd.DataFrame(df.pivot(index='cat', columns='date').rank(pct=True).diff(axis=1).std()).reset_index(level=1)
                rtrn_df.columns = ['Date', 'Metric Turnover']
                rtrn_df = rtrn_df.dropna()
                return rtrn_df
            except ValueError:
                return pd.DataFrame({'Date': [np.nan], 'Metric Turnover': [np.nan]})
        else:
            try:
                return df.pivot(index='cat', columns='date').rank(pct=True).diff(axis=1).std().mean()
            except:
                return np.nan

    def create_performance_differential(self, df):
        """Adds a 'performance diferential' to a provided pd.DataFrame."""

        differentials = ['Differential']
        for col in df.columns[1:]:
            try:
                d = ((df[col][0] * 2 + df[col][1]) / 3) - ((df[col][4] * 2 + df[col][3]) / 3)
            except (IndexError, KeyError) as e:
                d = np.nan
            differentials.append(d)
            QApplication.processEvents()
        df.loc[len(df)] = differentials
        return df
        
    def create_performance_differential_binary(self, df):
        """Adds a 'performance diferential' to a provided pd.DataFrame, for a binary metric."""

        differentials = ['Differential']
        for col in df.columns[1:]:
            try:
                d = df[col][0] - df[col][1]
            except (IndexError, KeyError) as e:
                d = np.nan
            differentials.append(d)
            QApplication.processEvents()
        df.loc[len(df)] = differentials
        return df

    def calculate_performance_binary(self, x, y, group_col=None, return_group_cols=True):
        """Calculates performance for a given binary x column
        
        Calculates performance for a given binary x column, where 'performance' 
        means the average y for each unique x value, in each unique date value, 
        if a group_col is provided.

        Parameters
        ----------
        x : pd.Series
            A pd.Series of the x column, the values to be evaluated
        y : pd.Series
            A pd.Series of the y column, the values to be averaged 
            for each unique x value
        group_col : pd.Series, optional
            A pd.Series of the column to group data by. These values are used 
            to slice the data before performing the above calculations.
        return_group_cols : bool, optional
            If True and if a group_col is provided, the performance is 
            calculated by group, and a column for each group is added 
            to the return pd.DataFrame

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame of the resulting performance calculations.
            If group_col is provided and return_group_cols is True, 
            the returned pd.DataFrame will include columns for each 
            unique value in group_col.
        """

        if len(x) == 0 or len(y) == 0:
            return pd.DataFrame({})

        bin_vals = x.dropna().unique()
        bin_vals.sort()
        bin_vals = bin_vals[::-1]

        if len(bin_vals) != 2:
            return pd.DataFrame({})

        rtn_df = pd.DataFrame({'Binary Value': bin_vals})

        if group_col is None:
            df = pd.DataFrame({'x': x, 'y': y}).dropna()
        else:
            df = pd.DataFrame({'x': x, 'y': y, 'group': group_col}).dropna()

        val1 = df.loc[df['x'] == bin_vals[0]]['y'].mean()
        val2 = df.loc[df['x'] == bin_vals[1]]['y'].mean()
        rtn_df['All'] = [val1, val2]

        if group_col is not None and return_group_cols:
            for group in group_col.sort_values().unique():
                if isinstance(group, str):
                    group_str = group
                elif isinstance(group, dt.datetime):
                    group_str = group.strftime("%Y-%m-%d")
                else:
                    group_str = str(group)
                group_df = df.loc[df['group'] == group]
                val1 = group_df.loc[group_df['x'] == bin_vals[0]]['y'].mean()
                val2 = group_df.loc[group_df['x'] == bin_vals[1]]['y'].mean()
                rtn_df[group_str] = [val1, val2]
                QApplication.processEvents()

        rtn_df = self.create_performance_differential_binary(rtn_df)
        return rtn_df

    def calculate_performance(self, x, y, group_col=None, return_group_cols=True):
        """Calculates performance for a given x column
        
        Calculates performance for a given x column, where 'performance' 
        means the average y for each quintile of x value, in each unique 
        date value, if a group_col is provided.

        Parameters
        ----------
        x : pd.Series
            A pd.Series of the x column, the values to be evaluated
        y : pd.Series
            A pd.Series of the y column, the values to be averaged 
            for each unique x value
        group_col : pd.Series, optional
            A pd.Series of the column to group by. These values are used 
            to slice the data before performing the above calculations.
        return_group_cols : bool, optional
            If True and if a group_col is provided, the performance is 
            calculated by group, and a column for each group is added 
            to the return pd.DataFrame

        Returns
        -------
        pd.DataFrame
            A pd.DataFrame of the resulting performance calculations.
            If group_col is provided and return_group_cols is True, 
            the returned pd.DataFrame will include columns for each 
            unique value in group_col.
        """
        if len(x) == 0 or len(y) == 0:
            return pd.DataFrame({"Quintile": [], "All": []})

        if len(x.dropna().unique()) == 2:
            return self.calculate_performance_binary(x,y,group_col,return_group_cols)

        if group_col is None:
            data_df = pd.DataFrame({'x': x, 'y': y})
            data_df['Quintile'] = create_quintile(x)
            rtn_df = data_df.groupby('Quintile').mean()['y'].reset_index().rename(columns={"y": "All"})

        else:
            def get_quints(vector):
                return create_quintile(vector)

            date_group = NumpyGrouper(group_col)
            quints = date_group.apply(get_quints, x)
            data_df = pd.DataFrame({'group': group_col, 'x': x, 'y': y, 'Quintile': quints})

            rtn_df = data_df.groupby('Quintile').mean()['y'].reset_index().rename(columns={"y": "All"})

            for quint in range(1, 6):
                if quint not in quints:
                    rtn_df.loc[len(rtn_df)] = [quint, np.nan]
            rtn_df = rtn_df.sort_values('Quintile')

            if return_group_cols:
                def add_group_performance(df):
                    try:
                        if isinstance(df.name, str):
                            group_str = df.name
                        elif isinstance(df.name, dt.datetime):
                            group_str = df.name.strftime('%Y-%m-%d')
                        else:
                            group_str = str(df.name)
                    except Exception as e:
                        print(df.name)
                        print(str(e))
                        return
                    rtn_df[group_str] = df.groupby('Quintile').mean().reset_index()['y']
                    QApplication.processEvents()
    
                data_df.groupby('group').apply(add_group_performance)
        rtn_df = self.create_performance_differential(rtn_df)

        return rtn_df

    @pyqtSlot(str)
    @handle_error(callback_attributes=['show_error', 'finish_status_task', 'finish_metric_dive_3d_standalone'], msg="Unknown error when running 3d plot. Please try another metric.")
    def run_metric_3d_standalone(self, z_col):

        if not self.data_is_loaded:
            self.finish_metric_dive_3d_standalone()
            return

        if not isinstance(z_col, str) or z_col == "":
            self.finish_metric_dive_3d_standalone()
            return

        self.metric_dive_cols["z_col"] = z_col
        metric_df = self.get_metric_df(include_cols=[z_col])

        if metric_df.empty:
            self.finish_metric_dive_3d_standalone()
            return

        if z_col not in metric_df.columns:
            self.finish_metric_dive_3d_standalone()
            return
        
        self.metric_dive_started_signal.emit()
        self.metric_dive_3d_signal.emit(metric_df, self.metric_dive_cols)
        self.finish_metric_dive_3d_standalone()

        return

    @pyqtSlot(dict, bool)
    @handle_error(callback_attributes=['show_error', 'finish_status_task', 'finish_metric_dive'], msg="Unknown error when running metric dive. Please try another metric.")
    def run_metric_dive(self, cols=None, remove_filters=False):
        """Run various calculations for a particular x column
        
        Run a slew of calculations as a deep dive into one particular 
        column. This includes, but not is limited to: metric turnover, 
        performance, sensitivity/specificity, qq plots, etc.

        Parameters
        ----------
        x_col : str
            A str which is the name of the column to be analyzed

        cat_col : str, optional
            A str which is the name of the column to be used as a
            category column in the analysis
        """

        if not self.data_is_loaded:
            self.finish_metric_dive()
            return

        if cols is None:
            cols = self.metric_dive_cols

        if not isinstance(cols["x_col"], str) or cols["x_col"] == "":
            self.finish_metric_dive()
            return

        if not isinstance(cols["y_col"], str) or cols["y_col"] == "":
            self.finish_metric_dive()
            return

        self.metric_dive_cols = cols

        if remove_filters:
            self.remove_metric_dive_filters(False)
            
        metric_df = self.get_metric_df()

        if metric_df.empty:
            self.show_error("Unknown error while running metric dive.")
            self.finish_metric_dive()
            return

        if self.metric_dive_x_col not in metric_df.columns or self.metric_dive_y_col not in metric_df.columns:
            self.show_error("One or more selected columns not valid.")
            self.finish_metric_dive()
            return

        self.metric_dive_started_signal.emit()


        metric_dive_outputs = {}
        self.create_and_emit_metric_dive_df(metric_df)
        self.create_and_emit_metric_dive_anova_stats(metric_df)
        self.create_and_emit_metric_dive_summary_stats(metric_df)
        self.create_and_emit_metric_dive_perf_date_table(metric_df)
        self.create_and_emit_metric_dive_perf_category_table(metric_df)
        self.create_and_emit_metric_dive_by_date(metric_df)
        self.create_and_emit_qq_norm(metric_df)
        self.create_and_emit_qq_y(metric_df)
        self.create_and_emit_metric_dive_dp_by_category(metric_df)
        self.create_and_emit_metric_dive_metric_turnover(metric_df)
        self.create_and_emit_metric_dive_sens_spec(metric_df)
        self.create_and_emit_metric_dive_box_and_whisker(metric_df)

        self.metric_dive_3d_signal.emit(metric_df, self.metric_dive_cols)

        self.finish_metric_dive()
        self.update_metric_dive_preview_model()

    def finish_metric_dive_3d_standalone(self, *args, **kwargs):
        self.metric_dive_3d_standalone_complete_signal.emit()

    def finish_metric_dive(self, *args, **kwargs):
        self.metric_dive_complete_signal.emit()

    def create_and_emit_qq_norm(self, df):
        """
        Calculate and emit the data for a QQ plot of the current metric dive 
        column's distribution against a theoretical normal distribution.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        x = pd.to_numeric(df[self.metric_dive_x_col], errors="coerce").dropna()
        sorted_x = list(x.sort_values())

        mu, s, n = x.mean(), x.std(), x.count()

        sorted_norms = []
        for i in range(n):
            if i+1 == n:
                p = .99
            else:
                p = (i+1)/n
            sorted_norms.append(norm.ppf(p, mu, s))
            QApplication.processEvents()

        sorted_df = pd.DataFrame({self.metric_dive_x_col: sorted_x, 'Normal': sorted_norms})
        #sorted_df.dropna(inplace=True)
        self.metric_dive_qq_norm.emit(sorted_df)

    def create_and_emit_qq_y(self, df):
        """
        Calculate and emit the data for a QQ plot of the current metric dive 
        column's distribution against the 'y' column's distribution.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        xy_df = df[[self.metric_dive_x_col, self.metric_dive_y_col]]
        xy_df.dropna(inplace=True)

        sorted_x = list(xy_df[self.metric_dive_x_col].sort_values())
        sorted_y = list(xy_df[self.metric_dive_y_col].sort_values())

        sorted_df = pd.DataFrame({self.metric_dive_x_col: sorted_x, self.metric_dive_y_col: sorted_y})
        self.metric_dive_qq_y.emit(sorted_df)

    def create_and_emit_metric_dive_metric_turnover(self, df):
        """
        Calculate and emit the metric turnover data for the current metric 
        dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        if not self.metric_dive_by_category or not self.metric_dive_by_date:
            self.metric_dive_metric_turnover.emit(None)
            return

        metric_df = df
        metric_turnover_df = self.calculate_metric_turnover(metric_df[self.metric_dive_x_col], metric_df[self.metric_dive_date_col], metric_df[self.metric_dive_category_col], include_df=True)
        metric_turnover_df['Date'] = pd.to_datetime(metric_turnover_df['Date']).dt.strftime("%Y-%m-%d")
        self.metric_dive_metric_turnover.emit(metric_turnover_df)

    def create_and_emit_metric_dive_by_date(self, df):
        """
        Calculate and emit correlation and number of data points by date for the 
        current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        if not self.metric_dive_by_date:
            self.metric_dive_date_df = pd.DataFrame({'Date': [], 'Data Points': [], 'Correlation': []})
            self.metric_dive_dp_by_date.emit(self.metric_dive_date_df)
            return

        metric_df = df
        dates     = []
        dps       = []
        correls   = []
        for date in metric_df[self.metric_dive_date_col].sort_values().unique():
            date_df = metric_df[metric_df[self.metric_dive_date_col] == date]

            dates.append(np.datetime_as_string(date, unit='D'))
            x_count = date_df[self.metric_dive_x_col].dropna().count()
            dps.append(x_count)

            clean_date_df = date_df[[self.metric_dive_x_col, self.metric_dive_y_col]].replace([np.inf, -np.inf], np.nan).dropna()
            xy_count = date_df[[self.metric_dive_x_col, self.metric_dive_date_col]].dropna().shape[0]
            if xy_count < 10:
                correls.append(np.nan)
            else:
                r_value, _ = pearsonr(clean_date_df[self.metric_dive_x_col], clean_date_df[self.metric_dive_y_col])
                correls.append(r_value)
            QApplication.processEvents()

        self.metric_dive_date_df = pd.DataFrame({'Date': dates, 'Data Points': dps, 'Correlation': correls})
        self.metric_dive_dp_by_date.emit(self.metric_dive_date_df)


    def create_and_emit_metric_dive_dp_by_category(self, df):
        """
        Calculate and emit number of data points by category for the 
        current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        if not self.metric_dive_by_category:
            self.metric_dive_dp_by_category.emit(None)
            return

        metric_df  = df
        categories = []
        dps        = []
        for category in metric_df[self.metric_dive_category_col].sort_values().unique():
            categories.append(category)
            dps.append(metric_df[metric_df[self.metric_dive_category_col] == category][self.metric_dive_category_col].count())
            QApplication.processEvents()

        self.metric_dive_dp_by_category.emit(pd.DataFrame({'Category': categories, 'Data Points': dps}))

    def create_and_emit_metric_dive_perf_category_table(self, df):
        """Calculate and emit performance (by category) for the current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        if not self.metric_dive_by_category:
            self.metric_dive_perf_category_df = pd.DataFrame({"Quintile": [], "All": []})
            self.metric_dive_perf_category_table_signal.emit(None)
            return

        self.metric_dive_perf_category_df = self.calculate_performance(df[self.metric_dive_x_col], df[self.metric_dive_y_col], group_col=df[self.metric_dive_category_col], return_group_cols=True)

        format_cols = {}
        for i in range(1, self.metric_dive_perf_category_df.shape[1]):
            format_cols[i] = (True, True)

        perf_model = NumpyModel(self.metric_dive_perf_category_df, format_cols=format_cols)

        self.metric_dive_perf_category_table_signal.emit(perf_model)

    def create_and_emit_metric_dive_perf_date_table(self, df):
        """Calculate and emit performance (by date) for the current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        if not self.metric_dive_by_date:
            self.metric_dive_perf_date_df = pd.DataFrame({"Quintile": [], "All": []})
            self.metric_dive_perf_date_table_signal.emit(None)
            return

        self.metric_dive_perf_date_df = self.calculate_performance(df[self.metric_dive_x_col], df[self.metric_dive_y_col], group_col=df[self.metric_dive_date_col], return_group_cols=True)

        format_cols = {}
        for i in range(1, self.metric_dive_perf_date_df.shape[1]):
            format_cols[i] = (True, True)

        perf_model = NumpyModel(self.metric_dive_perf_date_df, format_cols=format_cols)

        self.metric_dive_perf_date_table_signal.emit(perf_model)

    def create_and_emit_metric_dive_summary_stats(self, df):
        """Calculate and emit high level summary statistics for current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        cols = {}
        for metric in [self.metric_dive_x_col, self.metric_dive_y_col]:
            letter = "x_" if metric == self.metric_dive_x_col else "y_"
            series = pd.to_numeric(df[metric], errors="coerce")
            series.dropna(inplace=True)

            cols[letter+"name"]   = metric
            cols[letter+"dp"]     = series.size
            cols[letter+"std"]    = series.std()
            cols[letter+"mean"]   = series.mean()
            cols[letter+"median"] = series.median()
            cols[letter+"max"]    = series.max()
            cols[letter+"min"]    = series.min()
            QApplication.processEvents()

        self.metric_dive_summary_stats.emit(cols)

    def create_and_emit_metric_dive_box_and_whisker(self, df):
        """Calculate and emit data necessary for box and whisker plot 
        for current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        df = df
        self.metric_dive_box_and_whisker.emit([df[self.metric_dive_x_col]])

    def create_and_emit_metric_dive_anova_stats(self, df):
        """
        Calculate and emit anova stats for current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        metric_df = df

        count = metric_df.shape[0]
        x = metric_df[self.metric_dive_x_col]
        y = metric_df[self.metric_dive_y_col]

        if count >= 10:
            slope, intercept, r_value, p_value, std_err = linregress(x,y)
            r_squared = r_value*r_value
        else:
            slope, intercept, r_value, p_value, std_err = ["--"] * 5
            r_squared = "--"

        self.metric_dive_anova_dict = {
            "dp": count,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "correlation": r_value,
            "p_value": p_value,
            "std_err": std_err
        }

        self.metric_dive_anova_stats.emit(self.metric_dive_anova_dict)

    def create_and_emit_metric_dive_sens_spec(self, df):
        """
        Calculate and emit data necessary for sensitivity/specificity 
        plot for current metric dive column.

        Parameters
        ----------
        df : pd.DataFrame
            a pd.DataFrame with the data to be used in the calculations.
        """

        metric_df = df
        x = metric_df[self.metric_dive_x_col]
        y = metric_df[self.metric_dive_y_col]

        if not is_binary(x) or not is_binary(y):
            self.metric_dive_sens_spec.emit(None, None)
            return

        x_vals = x.dropna().unique()
        x_vals.sort()

        y_vals = y.dropna().unique()
        y_vals.sort()

        xy_df = pd.DataFrame({'x': x, 'y': y})
        xy_df = xy_df.dropna()

        total_cnt = xy_df.shape[0]

        if total_cnt == 0:
            self.metric_dive_sens_spec.emit(None, None)
            return

        true_neg  = xy_df.loc[(xy_df.x == x_vals[0]) & (xy_df.y == y_vals[0])].shape[0]
        false_neg = xy_df.loc[(xy_df.x == x_vals[0]) & (xy_df.y == y_vals[1])].shape[0]
        true_pos  = xy_df.loc[(xy_df.x == x_vals[1]) & (xy_df.y == y_vals[1])].shape[0]
        false_pos = xy_df.loc[(xy_df.x == x_vals[1]) & (xy_df.y == y_vals[0])].shape[0]

        rtn_df = pd.DataFrame({
            'Category': ['True Negative', 'False Negative', 'True Positive', 'False Positive'],
            'Count': [true_neg, false_neg, true_pos, false_pos],
            'x': [-1, -1, 1, 1],
            'y': [-1, 1, 1, -1],
            'x_label': [x_vals[0], x_vals[0], x_vals[1], x_vals[1]],
            'y_label': [y_vals[0], y_vals[1], y_vals[1], y_vals[0]]
        })

        size_dict = {
            'True Negative': true_neg / total_cnt,
            'False Negative': false_neg / total_cnt,
            'True Positive': true_pos / total_cnt,
            'False Positive': false_pos / total_cnt
        }

        self.metric_dive_sens_spec.emit(rtn_df, size_dict)

    def create_metric_dive_filter(self, filter_):
        """Create the given metric dive filter
        
        Creating a filter means adding a list-like of bools to a 
        class variable dict, which is used in compile_filters() 
        when accessing self.metric_dive_df.

        After creating a filter, run_metric_dive() is re-run, 
        to reflect the new changes.

        Parameters
        ----------
        filter_ : MetricDiveFilter
            The MetricDiveFilter to be created.
        """

        self.start_status_task_signal.emit("Applying metric dive filter: " + filter_.name)
        self.metric_dive_started_signal.emit()

        self.metric_dive_filters[filter_] = filter_.create_filter(self.get_metric_df(apply_filter=False, drop_na=False), cols=self.metric_dive_cols)
        self.update_metric_dive_preview_model()
        self.run_metric_dive(remove_filters=False)

    def remove_metric_dive_filters(self, rerun=True):
        """Removes all metric dive filters
        
        Removes all filters from class variable dict self.metric_dive_filters 
        which is used in compile_filters() when accessing self.metric_dive_df.

        Parameters
        ----------
        rerun : bool
            If true, run_metric_dive() is run to reflect the new changes
        """

        if bool(self.metric_dive_filters):
            self.metric_dive_filters = {}
            if rerun:
                self.run_metric_dive(remove_filters=False)
        else:
            self.metric_dive_filters = {}

    def create_and_emit_metric_dive_df(self, df):
        # Emit DF and column names
        self.metric_dive_df_signal.emit(df, self.metric_dive_cols)

    @pyqtSlot(object)
    @handle_error(callback_attributes=['show_error', 'finish_status_task'], msg_prefix="Error when creating transformation. '", msg_suffix="' Please try something else.")
    def create_transformation(self, transformation):
        """Create new columns based on the given Transformation
        
        Creating a transformation means adding new columns according to 
        the logic of the given Transformation. 

        Parameters
        ----------
        transformation : Transformation
            The Transformation to be used to create the new columns
        """

        base_df = self.get_current_df(apply_filter=False, clean=False)

        filter_idx = self.compile_filters(base_df)

        transform_cols = transformation.create_transformation(base_df[filter_idx].reset_index(drop=True))
        for key in transform_cols:
            col = transform_cols[key]

            if col is None or len(col) != sum(filter_idx):
                raise Exception('Something went wrong with the transformation ' + transformation.name + '. Please check your inputs and try again.')
            
            s             = pd.Series([np.nan] * len(filter_idx))
            s[filter_idx] = list(col)
            base_df[key]  = s

        self.set_current_df(base_df)

        self.transformations.append(transformation)
        self.update_data_preview_model()
        self.update_cols_signal.emit(list(base_df.columns), list(self.x_cols))
        self.finish_status_task()
        self.transformation_completed_signal.emit(transformation)

    def remove_transformation(self, transformation):
        """Remove the columns created by the given Transformation

        Transformations keep a record of the ids of the columns they created. 
        This method removes those columns from the current dataset.

        Parameters
        ----------
        transformation : Transformation
            The transformation to be removed
        """

        current_df = self.get_current_df(apply_filter=False, clean=False)
        for col in transformation.col_ids:
            if col in self.base_df.columns:
                current_df.drop(col, axis=1, inplace=True)
        self.set_current_df(current_df)

        if transformation in self.transformations:
            self.transformations.remove(transformation)
        self.update_data_preview_model()
        self.update_cols_signal.emit(list(self.get_current_df().columns), list(self.x_cols))
    
    @handle_error(callback_attributes=['show_error', 'finish_status_task'], msg_prefix="Error when creating filter. '", msg_suffix="' Please try something else.")
    def create_filter(self, filter_):
        """Create the given filter  
        
        Creating a filter means adding a list-like of bools to a 
        class variable dict, which is used in compile_filters() 
        when accessing the current dataframe.

        Parameters
        ----------
        filter_ : Filter
            The Filter to be created.
        """
        df = self.get_current_df(apply_filter=False, clean=False)

        self.filters[filter_] = filter_.create_filter(df)
        self.filters_change_signal.emit(list(self.filters.keys()))
        self.update_data_preview_model()
        self.finish_status_task()
        self.filter_applied_signal.emit(filter_)

    def remove_filter(self, filter_):
        """Remove the given filter
        
        Because filters are applied each time the dataframe is accessed, 
        removing filters simply means removing them from the class dict 
        that holds the current filters.
        """

        self.filters.pop(filter_, None)
        self.filters_change_signal.emit(list(self.filters.keys()))
        self.update_data_preview_model()

    def remove_all_filters(self):
        """Remove all filters
        
        Because filters are applied each time the dataframe is accessed, 
        removing filters simply means clearing the dict that holds them.
        """

        self.filters = {}
        self.filters_change_signal.emit(list(self.filters.keys()))

    @pyqtSlot(dict)
    def aggregate_by_date(self, info):
        """Aggregate the current dataset by date
        
        Given a certain date format (for example: '%Y' indicating annual), 
        aggregate rows in the dataset down to that level of detail, 
        applying a user-provided function to each column, such as Sum or Mean.

        Parameters
        ----------
        info : dict
            A dict holding relevant information, like aggregation function, 
            date aggregation level, category column (optional), etc. 
        """

        self.start_status_task_signal.emit("Aggregating by date...")
        self.remove_all_filters()
        df = self.get_current_df(apply_filter=False, clean=False).copy()

        def agg_cols(df, ignore_cols):
            def agg_col(s, ignore_cols):
                if s.name in ignore_cols:
                    return s.unique()[0]
                n = pd.to_numeric(s, errors='coerce').dropna()

                if n.size == 0:
                    return np.nan
                agg_func = info['agg_func']
                if   agg_func == "Sum":
                    return n.sum()
                elif agg_func == "Mean":
                    return n.mean()
                elif agg_func == "Median":
                    return n.median()
                elif agg_func == "Mode":
                    return n.mode()
                elif agg_func == "Max":
                    return n.max()
                elif agg_func == "Min":
                    return n.min()

            df = df.apply(lambda s: agg_col(s, ignore_cols))
            return df

        try:
            df[info['date_col']] = pd.to_datetime(df[info['date_col']])
        except:
            self.show_error('Date column cannot be read as a date. Make sure dates are in a proper format, and that there are no missing values. https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior')
            return

        df[info['date_col']] = df[info['date_col']].dt.strftime(info['agg_level'])

        if info['category_col'] in df.columns:
            group_cols = [info['date_col'], info['category_col']]
            #df = df.groupby(group_cols).apply(lambda _df: agg_cols(_df, info['agg_func'], group_cols))
        else:
            group_cols = [info['date_col']]
            #df = df.groupby(info['date_col']).apply(lambda _df: agg_cols(_df, info['agg_func'], [info['date_col']]))

        df = df.groupby(group_cols).apply(lambda df: agg_cols(df, group_cols))
        for col in group_cols:
            try:
                df.drop(col, axis=1, inplace=True)
            except KeyError:
                pass
        df = df.reset_index()

        self.date_agg_df = df

        self.update_data_preview_model()
        self.finish_status_task()
        self.date_aggregation_complete_signal.emit()

    @pyqtSlot()
    def remove_date_aggregation(self):
        """Remove the date aggregation applied to the dataset."""

        self.start_status_task_signal.emit("Removing date aggregation...")
        self.date_agg_df = None
        self.remove_all_filters()
        self.update_data_preview_model()
        self.remove_date_aggregation_complete_signal.emit()
        self.finish_status_task()

    def compile_metric_dive_filters(self, df):
        """Compile the current metric dive filters and return a list of bools
        
        This method takes all of the current metric dive filters and creates 
        a single list of bools, indicating which rows of the current metric 
        dive dataframe should be kept.

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame that the filters will be applied to. This is only
            used to determine the length of the list of True, if there are no 
            filters to be applied.

        Returns
        -------
        list
            A list of bools, indicating which rows of the metric dive dataframe 
            should be kept.
        """
        
        keys = list(self.metric_dive_filters.keys())
        if len(keys) == 0:
            return [True] * df.shape[0]
        bools = self.metric_dive_filters[keys[0]]
        for key in keys:
            bools = list(map(all, zip(bools, self.metric_dive_filters[key])))
        return bools

    def compile_filters(self, df):
        """Compile the current filters and return a list of bools
        
        This method takes all of the current filters and creates 
        a single list of bools, indicating which rows of the current dataframe 
        should be kept.

        Parameters
        ----------
        df : pd.DataFrame
            A pd.DataFrame that the filters will be applied to. This is only
            used to determine the length of the list of True, if there are no 
            filters to be applied.

        Returns
        -------
        list
            A list of bools, indicating which rows of the current dataframe 
            should be kept.
        """

        keys = list(self.filters.keys())
        if len(keys) == 0:
            return [True] * df.shape[0]
        bools = self.filters[keys[0]]
        for key in keys:
            bools = list(map(all, zip(bools, self.filters[key])))
        return bools
    
    def update_data_preview_model(self):
        """Create and emit a model of the current dataset, to be used in views."""

        dfmodel = NumpyModel(self.get_current_df(apply_filter=True, clean=False))
        self.data_preview_model_signal.emit(dfmodel)

    def update_metric_dive_preview_model(self):
        """Create and emit a model of the current metric dive dataset, to be used in views."""

        dfmodel = NumpyModel(self.get_metric_df(date_as_str=True))
        self.metric_dive_data_preview_model_signal.emit(dfmodel)

    @pyqtSlot(str)
    def prepare_metric_dive_for_report(self, file_nm):
        """Emit data and information necessary to create the metric dive report."""

        metric_df = self.get_metric_df()
        if metric_df.empty:
            return

        if self.metric_dive_y_col == "":
            return

        if self.metric_dive_x_col == "":
            return

        cols = {
            "x": self.metric_dive_x_col,
            "y": self.metric_dive_y_col,
            "date": self.metric_dive_date_col,
            "category": self.metric_dive_category_col
        }
        self.metric_dive_report_signal.emit(metric_df, self.metric_dive_perf_date_df, self.metric_dive_date_df, self.metric_dive_anova_dict, cols, file_nm)

    @pyqtSlot(str)
    def save_df(self, file_nm):
        """Save current dataset to the provided location."""

        if not file_nm.endswith(".csv"):
            file_nm = file_nm+".csv"
        self.get_current_df(apply_filter=True, clean=False).to_csv(file_nm, index=False)


class ReportWorker(Worker):

    start_status_task_signal = pyqtSignal(str)
    finish_status_task_signal = pyqtSignal()

    DEFAULT_TOOLS  = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    PLOT_WIDTH     = 1700
    PLOT_HEIGHT    = 800
    STRETCH_POLICY = "stretch_both"

    def __init__(self, palette):
        super().__init__()
        self.palette = palette

    @pyqtSlot(pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, str)
    def generate_report(self, df, perf_df, date_df, anova_dict, cols, file_nm):
        self.start_status_task_signal.emit('Saving metric report...')

        summary_tab    = self.create_summary_tables(perf_df, date_df, anova_dict)
        df_tab         = self.create_data_table(df, cols)
        scatter_tab    = self.create_scatter_tab(df, cols)
        perf_tab       = self.create_perf_tab(perf_df)
        correl_tab     = self.create_correlation_by_date(date_df)
        dp_tab         = self.create_data_points_by_date(date_df)
        hist_tab       = self.create_histogram(df[cols['x']])
        qq_tab         = self.create_qq_tab(df, cols)

        # Piece together tabs
        tabs = Tabs(tabs = [summary_tab, df_tab, scatter_tab, perf_tab, correl_tab, dp_tab, hist_tab, qq_tab])

        # Header
        header = Div(text='<h1>X Column: <b>' + cols['x'] + '&nbsp;&nbsp;&nbsp;&nbsp;</b>Y Column: <b>' + cols['y'] + '</b></h1>')

        # Root
        root = column(header, tabs)

        # Save document
        curdoc().clear()
        curdoc().add_root(root)

        save(curdoc(), file_nm, title=cols['x'] + ' ~ ' + cols['y'])

        self.finish_status_task_signal.emit()

    def create_summary_tables(self, perf_df, date_df, anova_dict):
        date_df_copy = date_df.copy()
        ###############################
        # Correlation Periods Summary #
        ###############################
        tot_periods = date_df_copy['Correlation'].dropna().shape[0]
        pos_periods = sum([1 for val in date_df_copy['Correlation'] if try_float(val) > 0])
        try:
            pos_percent = (pos_periods / tot_periods) * 100
        except ZeroDivisionError:
            pos_percent = np.nan

        neg_periods = sum([1 for val in date_df_copy['Correlation'] if try_float(val) < 0])
        try:
            neg_percent = (neg_periods / tot_periods) * 100
        except ZeroDivisionError:
            neg_percent = np.nan

        # Weighted Average
        date_df_copy['CorrelxDP'] = date_df_copy['Data Points'] * date_df_copy['Correlation']
        date_df_copy = date_df_copy.dropna()
        try:
            wghtd_avg = sum(date_df_copy['CorrelxDP']) / sum(date_df_copy['Data Points'])
        except ZeroDivisionError:
            wghtd_avg = np.nan

        correl_summary_df = pd.DataFrame({
            'Statistic': ['Total Periods', 'Negative Periods', 'Positive Periods', '% Negative', '% Positive', 'Weighted Average Correlation'],
            'Value': [round_default(tot_periods), round_default(neg_periods), round_default(pos_periods), str(round_default(neg_percent))+'%', str(round_default(pos_percent))+'%', round_default(wghtd_avg)]
        })
        correl_summary_source  = ColumnDataSource(correl_summary_df)
        correl_summary_columns = [TableColumn(field='Statistic', title='Statistic'), TableColumn(field='Value', title='Value')]
        correl_summary_table   = DataTable(source=correl_summary_source, columns=correl_summary_columns, index_position=None)
        correl_summary_column  = column(Div(text='<h2 style="text-align: center;">Correlation Summary</h2>'), correl_summary_table)

        #####################
        # Performance Table #
        #####################
        perf_source  = ColumnDataSource(perf_df[perf_df.columns[:2]])
        perf_columns = [TableColumn(field=perf_df.columns[0], title=perf_df.columns[0]), TableColumn(field='All', title='All')]
        perf_table   = DataTable(source=perf_source, columns=perf_columns, index_position=None)
        perf_column  = column(Div(text='<h2 style="text-align: center;">Performance by metric quintile (or binary value)</h2>'), perf_table)

        #####################
        # Fit Summary Table #
        #####################
        anova_df = pd.DataFrame({
            'Statistic': ['R-Squared', 'P-Value', 'Intercept', 'Slope', 'Standard Error', 'Data Points'],
            'Value': [anova_dict['r_squared'], anova_dict['p_value'], anova_dict['intercept'], anova_dict['slope'], anova_dict['std_err'], anova_dict['dp']]
            })
        anova_source  = ColumnDataSource(anova_df)
        anova_columns = [TableColumn(field='Statistic', title='Statistic'), TableColumn(field='Value', title='Value')]
        anova_table   = DataTable(source=anova_source, columns=anova_columns, index_position=None)
        anova_column  = column(Div(text='<h2 style="text-align: center;">Fit Summary</h2>'), anova_table)

        tables_row = row(correl_summary_column, perf_column, anova_column)

        tables_tab = Panel(child=tables_row, title="Summary Tables")
        return tables_tab
        
    def create_data_table(self, df, cols):
        source = ColumnDataSource(df)
        datefmt = DateFormatter(format="%Y-%m-%d")
        columns = []
        for col in df.columns:
            if col == cols['date']:
                columns.append(TableColumn(field=col, title=col, formatter=datefmt))
            else:
                columns.append(TableColumn(field=col, title=col))
        df_table = DataTable(source=source, columns=columns, sizing_mode=self.STRETCH_POLICY, index_position=None)
        df_tab = Panel(child=df_table, title="Data")
        return df_tab

    def create_histogram(self, x):
        bins = math.ceil(x.dropna().count() ** 0.5)
        y,x = np.histogram(x, bins=bins)
        lefts  = x[:-1]
        rights = x[1:]

        hist_source = ColumnDataSource(pd.DataFrame({'left': lefts, 'right': rights, 'count': y}))
        hist_fig = figure(tools=self.DEFAULT_TOOLS, sizing_mode=self.STRETCH_POLICY)
        hist_fig.quad(bottom=0, left='left', right='right', top='count', source=hist_source)
        hist_tab = Panel(child=hist_fig, title="Histogram")
        return hist_tab

    def create_correlation_by_date(self, df):

        if df.empty:
            return Panel(child=figure(x_range=df['Date'], tools=self.DEFAULT_TOOLS, sizing_mode=self.STRETCH_POLICY), title="Correlation by Date")

        def label_color(row, col):
            try:
                x = float(row[col])
            except ValueError:
                return '#9b9b9b'

            if math.isnan(x):
                return '#9b9b9b'

            if x >= 0:
                return '#6fd36f'
            else:
                return '#e55751'
        
        df['__color_col__'] = df.apply(lambda row: label_color(row, 'Correlation'), axis=1)

        source = ColumnDataSource(df)
        correl_fig = figure(x_range=df['Date'], tools=self.DEFAULT_TOOLS, sizing_mode=self.STRETCH_POLICY)
        correl_fig.vbar(x='Date', top='Correlation', width=0.5, color='__color_col__', source=source)
        correl_fig.xaxis.major_label_orientation = "vertical"
        correl_tab = Panel(child=correl_fig, title="Correlation by Date")
        return correl_tab

    def create_data_points_by_date(self, df):

        if df.empty:
            return Panel(child=figure(x_range=df['Date'], tools=self.DEFAULT_TOOLS, sizing_mode=self.STRETCH_POLICY), title="Data Points by Date")

        def label_color(row, col):
            try:
                x = float(row[col])
            except ValueError:
                return '#9b9b9b'

            if math.isnan(x):
                return '#9b9b9b'

            if x >= 0:
                return '#6fd36f'
            else:
                return '#e55751'
            
        df['__color_col__'] = df.apply(lambda row: label_color(row, 'Correlation'), axis=1)

        source = ColumnDataSource(df)
        dp_fig = figure(x_range=df['Date'], tools=self.DEFAULT_TOOLS, sizing_mode=self.STRETCH_POLICY)
        dp_fig.vbar(x='Date', top='Data Points', width=0.5, color='__color_col__', source=source)
        dp_fig.xaxis.major_label_orientation = "vertical"
        dp_tab = Panel(child=dp_fig, title="Data Points by Date")
        return dp_tab

    def create_perf_tab(self, df):
        source = ColumnDataSource(df)
        columns = []
        for col in df.columns:
            columns.append(TableColumn(field=col, title=col))
        perf_table = DataTable(source=source, columns=columns, sizing_mode="stretch_both", index_position=None, fit_columns=False)
        perf_tab = Panel(child=perf_table, title="Performance")
        return perf_tab

    def create_qq_tab(self, df, cols):
        xy_df = df[[cols['x'], cols['y']]].dropna()
        sorted_x = list(xy_df[cols['x']].sort_values())
        sorted_y = list(xy_df[cols['y']].sort_values())

        xy_df_sorted = pd.DataFrame({cols['x']: sorted_x, cols['y']: sorted_y})
        source = ColumnDataSource(xy_df_sorted)
        qq_fig = figure(tools=self.DEFAULT_TOOLS, sizing_mode=self.STRETCH_POLICY)
        qq_fig.scatter(cols['x'], cols['y'], fill_color='#9b9b9b', line_color='black', fill_alpha=0.6, source=source)
        qq_tab = Panel(child=qq_fig, title="QQ - Y")
        return qq_tab

    def create_scatter_tab(self, df, cols):
        color_dict = {}
        if cols['category'] is not None and cols['category'] != "" and cols['category'] in df.columns:
            for i, c in enumerate(df[cols['category']].unique()):
                color_dict[c] = self.palette[i]

        def label_color(row, col, color_dict):
            return color_dict.get(row[col], self.palette[0])

        if cols["category"] in df.columns:
            df['__color_col__'] = df.apply(lambda row: label_color(row, cols['category'], color_dict), axis=1)
        else:
            df['__color_col__'] = self.palette[0]

        scatter_source = ColumnDataSource(df)
        x_col = cols['x']
        y_col = cols['y']
        category_col = cols['category']
        color_col = '__color_col__'

        # Scatter
        scatter_fig = figure(tools=self.DEFAULT_TOOLS, sizing_mode=self.STRETCH_POLICY)
        if cols['category'] in df.columns:
            scatter_fig.scatter(x_col, y_col, fill_color=color_col, line_color=None, fill_alpha=0.6, legend=category_col, source=scatter_source)
        else:
            scatter_fig.scatter(x_col, y_col, fill_color=color_col, line_color=None, fill_alpha=0.6, source=scatter_source)
        scatter_tab = Panel(child=scatter_fig, title="Scatter")

        return scatter_tab



class LoadCsvWorker(Worker):
    
    loaded_df_signal = pyqtSignal(pd.DataFrame, str)
    loaded_df_model_signal = pyqtSignal(object)
    loading_msg_signal = pyqtSignal(str)
    show_error_signal = pyqtSignal(str)
    finish_task_signal = pyqtSignal(object)

    def __init__(self, parent, fileName):
        super().__init__()
        self.parent   = parent
        self.fileName = fileName

    def show_error(self, msg=""):
        """Wrapper method for emitting signal containing a message describing an encountered problem."""

        self.show_error_signal.emit(msg)

    def finish_task(self, page=None):
        """Wrapper method for emitting signal indicating the current task has been completed."""

        self.finish_task_signal.emit(page)

    @pyqtSlot()
    def work(self):
        from pages import SplashPage
        df = None
        data = []
        row_limit = 50000
        starting_mem = psutil.virtual_memory().percent
        available_mem = 100-starting_mem
        with open(self.fileName, encoding="utf8") as f:
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

        if df is None or df.empty:
            self.show_error("Data set uploaded is empty. Please upload a CSV with data.")
            self.finish_task(SplashPage)
        else:
            self.loaded_df_signal.emit(df, self.fileName)
            dfmodel = NumpyModel(df)
            self.loaded_df_model_signal.emit(dfmodel)
        