import pandas as pd
import numpy as np
import plotly.graph_objs as go


def generate_box_dict(vals):
	box_dict            = {}
	box_dict['med']     = vals.median()
	box_dict['q1']      = vals.quantile(0.25)
	box_dict['q3']      = vals.quantile(0.75)
	box_dict['whislo']  = vals.min()
	box_dict['whishi']  = vals.max()

	return box_dict


def is_binary(series, remove_na=True):
	if remove_na:
		s = series.dropna()
	else:
		s = series
	vals = s.unique()

	if len(vals) == 2:
		return True
	else:
		return False


def is_number(num, convert=False):
	if convert:
		try:
			num = float(num)
		except ValueError:
			return False
	
	try:
		return not np.isnan(num) and not np.isinf(num)
	except TypeError:
		try:
			num = float(num)
		except ValueError:
			return False
		return not np.isnan(num) and not np.isinf(num)
		


def clean_string(string):
        cleaned = string.replace('!', '').replace('@', '').replace('#', '').replace('$', '').replace('%', '').replace('^', '')
        cleaned = cleaned.replace('&', '').replace('*', '').replace('(', '').replace(')', '').replace('-', '').replace('=', '')
        cleaned = cleaned.replace('+', '').replace('.', '')
        return cleaned


def try_float(val, default=np.nan):
	try:
		val_float = float(val)
	except (ValueError, TypeError):
		return default

	return val_float


def round_default(val, dec=2, default="--"):
	try:
		val_float = float(val)
	except ValueError:
		return default

	try:
		return round(val, dec)
	except TypeError:
		return default


def try_sort(df, sort_col, ascending=True):
	df_copy = df.copy()
	if sort_col in df_copy.columns:
		try:
			df_copy['__temp_date'] = pd.to_datetime(df_copy[sort_col])
		except:
			df_copy['__temp_date'] = df_copy[sort_col]

		df_copy = df_copy.sort_values(by='__temp_date', ascending=ascending)
		df_copy.drop('__temp_date', axis=1, inplace=True)
	return df_copy


def elide_text(text, max_length=30):
	text = str(text)
	if len(text) <= 30:
		return text

	return text[:15] + "..." + text[-15:]


def create_quintile(x):
	"""Return a list of quintiles for the provided list-like."""

	if isinstance(x, pd.Series):
		n = x.reset_index(drop=True)
		nas = list(n[n.isnull()].index)
	elif isinstance(x, (np.ndarray, np.generic)):
		nas = np.argwhere(np.isnan(x))

	n = pd.to_numeric(x, errors="coerce")
	q = np.nanpercentile(n, [20,40,60,80])
	q = list(5 - np.searchsorted(q, n))
	q = [x if i not in nas else np.nan for i, x in enumerate(q)]
	
	return q