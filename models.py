from PyQt5 import QtCore
from PyQt5 import QtGui
import pandas as pd
import numpy as np
from IPython import embed;

class NumpyModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None, format_cols={}):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = np.array(data.values)
        self._cols = data.columns
        self._format_cols = format_cols
        self._maxs = {}
        self._mins = {}
        self._medians = {}
        
        if self._data.size != 0:
            for col in self._format_cols.keys():
                self._maxs[col] = np.nanmax(pd.to_numeric(self._data[:,col], errors="coerce"))
                self._mins[col] = np.nanmin(pd.to_numeric(self._data[:,col], errors="coerce"))
                self._medians[col] = np.nanmedian(np.unique(pd.to_numeric(self._data[:,col], errors="coerce")))
        
        self.r, self.c = np.shape(self._data)

    def rowCount(self, parent=None):
        return self.r

    def columnCount(self, parent=None):
        return self.c

    def toFloatZero(self, num, default):
        try:
            flt = float(num)
        except ValueError:
            flt = default
        return(flt)
    
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return self._data[index.row(),index.column()]
            
            # Check if column should be conditionally formatted
            if index.column() in self._format_cols:

                num = self.toFloatZero(self._data[index.row(),index.column()], np.nan)
                col_min = self.toFloatZero(self._mins[index.column()], num)
                col_max = self.toFloatZero(self._maxs[index.column()], num)
                col_median = self.toFloatZero(self._medians[index.column()], num)

                zero_mid = self._format_cols[index.column()][0]
                higher_better = self._format_cols[index.column()][1]
                
                # Adjust so all figures are >= 0, to avoid calculation complications
                n_num, n_min, n_max, n_median, n_zero = [x-col_min for x in [num, col_min, col_max, col_median, 0]]
                
                # If column should be formatted with negatives,
                # use zero as the mid point; otherwise use the median.
                if zero_mid:
                    n_mid = n_zero
                else:
                    n_mid = n_median
                
                # If the number is above the mid point, color green,
                # and do the transparency calculation slightly differently
                if n_num > n_mid:
                    try:
                        transparency = 1 - ((n_num - n_mid) / (n_max - n_mid))
                    except ZeroDivisionError:
                        transparency = 1
                    if role ==  QtCore.Qt.BackgroundRole:
                        if higher_better:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(255*transparency, 255, 255*transparency, 255*(1-transparency))))
                        else:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(255, 255*transparency, 255*transparency, 255*(1-transparency))))
                    if role == QtCore.Qt.ForegroundRole:
                        if transparency > 0.6:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(180,180,180)))
                        else:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(0,0,0)))
                # If below the mid point, color red,
                # and do the transparency calculation slightly differently
                if n_num < n_mid:
                    try:
                        transparency = 1 - ((n_mid - n_num) / (n_mid - n_min))
                    except ZeroDivisionError:
                        transparency = 1
                    if role ==  QtCore.Qt.BackgroundRole:
                        if not higher_better:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(255*transparency, 255, 255*transparency, 255*(1-transparency))))
                        else:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(255, 255*transparency, 255*transparency, 255*(1-transparency))))
                    if role == QtCore.Qt.ForegroundRole:
                        if transparency > 0.6:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(180,180,180)))
                        else:
                            return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(0,0,0)))
                else:
                    if role ==  QtCore.Qt.BackgroundRole:
                        return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(255, 255, 255, 0)))
                    if role == QtCore.Qt.ForegroundRole:
                        return QtCore.QVariant(QtGui.QBrush(QtGui.QColor(180,180,180)))
        return None

    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        if order:
            self._data = self._data[self._data[:,column].argsort()[::-1]]
        else:
            self._data = self._data[self._data[:,column].argsort()]

        self.layoutChanged.emit()

    def headerData(self, p_int, orientation, role):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._cols[p_int]
            elif orientation == QtCore.Qt.Vertical:
                return p_int
        return None

class PandasModel(QtCore.QAbstractTableModel): 
    def __init__(self, df = pd.DataFrame(), parent=None): 
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError, ):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.ix[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()): 
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending= order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()