from tkinter import E
import numpy as np
import pandas as pd
import transformations
import filters
import random as rd
import math
import re
import os
import io
import csv
import IPython
import traceback
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

from util import round_default, generate_box_dict, elide_text
from data_objects import Coord
from handlers import PlotlyHandler, CallHandler
from models import NumericSortProxyModel, AccurateRowNumberProxy

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QSortFilterProxyModel, QRegExp, pyqtSignal, pyqtSlot, QObject, QSize, QUrl
from PyQt5.QtWidgets import QTabWidget, QApplication, QCompleter, QStackedWidget, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel, QTableView, QFileDialog, QCheckBox, QListWidget, QLineEdit, QLayout
from PyQt5.QtGui import QFont, QPainter, QBrush, QColor, QKeySequence
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel

import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import plotly.io as pio
import plotly.graph_objects as go


class WebView(QWebEngineView):
    instance_cnt = 0
    def __init__(self):
        super(WebView, self).__init__()
        self.id = str(WebView.instance_cnt)
        WebView.instance_cnt += 1
        self.srcs = ["<script src='qrc:///qtwebchannel/qwebchannel.js'></script>"]
        self.load_handler()
        self.load_channel()
        self.loadFinished.connect(self.on_load)

    def on_load(self):
        script = "new QWebChannel(qt.webChannelTransport, function (channel) {window.handler = channel.objects.handler;});"
        self.run_javascript(script)

    def load_handler(self):
        self.handler = CallHandler()

    def load_channel(self):
        self.channel = QWebChannel()
        self.channel.registerObject('handler', self.handler)
        self.page().setWebChannel(self.channel)

    def display_html(self, html):
        raw_html = '<html style="background: rgba(0,0,0,0);"><head><meta charset="utf-8" />'
        for src in self.srcs:
            raw_html += src
        raw_html += '</head>'
        raw_html += '<body>'
        raw_html += """
            <script language="JavaScript">
            new QWebChannel(qt.webChannelTransport, function (channel) {
                window.handler = channel.objects.handler;
            });
            </script>
        """
        raw_html += html
        raw_html += '</body></html>'
        self.setHtml(raw_html)

    def run_javascript(self, script, callback=None):
        if callable(callback):
            self.page().runJavaScript(script, callback)
        else:
            self.page().runJavaScript(script)

    def inject_srcs(self, file_nm):
        with open(file_nm, 'r') as file:
            data = file.read().replace('\n', '')

        with open(file_nm, 'w') as file:
            for src in self.srcs:
                loc = data.find('</head>')
                data = data[:loc] + src + data[loc:]
            file.write(data)
            file.truncate()

    def display_file(self, file_nm):
        file_path = os.path.abspath(os.path.join(os.path.dirname(''), file_nm))
        self.inject_srcs(file_path)
        local_url = QUrl.fromLocalFile(file_path)
        self.load(local_url)     


class PlotlyView(WebView):

    def __init__(self, controller):
        super(PlotlyView, self).__init__()
        self.srcs.append("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")
        self.controller = controller
        self.page().setBackgroundColor(QColor(Qt.transparent))
        self.set_formatting()
        self.uniform_marker_style = dict(color=self.controller.PLOT_PALETTE_HEX[0], line=dict(width=2, color='Black'))

    def load_handler(self):
        self.current_selection = []
        self.handler = PlotlyHandler()
        self.handler.set_on_select(self.on_select)
        self.handler.set_on_deselect(self.on_deselect)

    def on_load(self):
        super(PlotlyView, self).on_load()
        self.page().toHtml(self.register_events)

    def register_events(self, html):
        id_regex = plot_id = re.findall('(?<=div id=")[a-zA-Z0-9-]+(?=")', html)
        try:
            plot_id = id_regex[0]
        except IndexError:
            return

        js = f"var plot = document.getElementById('{plot_id}');"
        #js += f"handler.js_to_python('{plot_id}', function(retVal) {{}});"
        js += """
        plot.on('plotly_hover', function(data) {
            if (typeof data === 'undefined') return;
            data.points.map(function(d){
                handler.on_hover({'curve': d['curveNumber'], 'x': d['x'], 'y': d['y'], 'text': d['data']['text'], 'customdata': d['customdata'], 'index': d['pointIndex']});
            });
        });

        plot.on('plotly_unhover', function(data) {
            if (typeof data === 'undefined') return;
            data.points.map(function(d){
                handler.on_unhover({'curve': d['curveNumber'], 'x': d['x'], 'y': d['y'], 'text': d['data']['text'], 'customdata': d['customdata'], 'index': d['pointIndex']});
            });
        });

        plot.on('plotly_click', function(data) {
            if (typeof data === 'undefined') return;
            data.points.map(function(d){
                handler.on_click({'curve': d['curveNumber'], 'x': d['x'], 'y': d['y'], 'text': d['data']['text'], 'customdata': d['customdata'], 'index': d['pointIndex']});
            });
        });

        plot.on('plotly_selected', function(data) {
            if (typeof data === 'undefined'){
                Plotly.restyle(plot, 'selectedpoints', null);
                handler.on_select(null);
                return;
            }
            var selection = []
            data.points.map(function(d){
                selection.push({'curve': d['curveNumber'], 'x': d['x'], 'y': d['y'], 'text': d['data']['text'][d['pointIndex']], 'customdata': d['customdata'], 'index': d['pointIndex']});
            });
            handler.on_select(selection);
        });
        """
        self.run_javascript(js)

    def on_select(self, selection):
        self.current_selection = selection

    def on_deselect(self):
        self.current_selection = []

    def get_selected_points(self):
        return self.current_selection

    def set_formatting(self):

        self.template = go.layout.Template()
        scattergls = []
        scatters = []
        for color in self.controller.PLOT_PALETTE_HEX:
            scattergls.append(go.Scattergl(marker=dict(color=color)))

        self.template.data.scattergl = scattergls

        self.axis_format = {
            'tickfont': dict(color=self.controller.FONT_COLOR, size=18),
            'ticks': 'inside',
            'showgrid': True,
            'gridwidth': 1,
            'gridcolor': self.controller.BACKGROUND_PRIMARY,
            'showline': True,
            'linewidth': 2,
            'linecolor': self.controller.BACKGROUND_SECONDARY
        }

        self.layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,1)',
            plot_bgcolor='rgba(0,0,0,1)',
            margin={'l':25,'r':0,'b':25,'t':25,'pad':0},
            xaxis=self.axis_format,
            yaxis=self.axis_format,
            dragmode='lasso',
            template="plotly_dark"
        )

        self.config = {
            'displaylogo': False
        }

    def scatter(self, df, x_col, y_col, category_col=None, date_col=None, add_trendline=False):
        traces = []
        
        df = df.copy()
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

        if df[x_col].dropna().shape[0] == 0 or df[y_col].dropna().shape[0] == 0:
            return

        if category_col is not None and category_col in df.columns:
            def create_traces(grp):
                hover_template = '<b>'+x_col+'</b>: %{x}<br><b>' + y_col+'</b>: %{y}<br>' + '<b>' + category_col+'</b>: '+str(grp.name)
                if date_col is not None and date_col in df.columns:
                    hover_template += '<br><b>'+date_col+'</b>: %{customdata}'
                    trace = go.Scattergl(mode='markers', y=grp[y_col], x=grp[x_col], text=grp.index, customdata=grp[date_col], name=str(grp.name), hovertemplate=hover_template)
                else:
                    trace = go.Scattergl(mode='markers', y=grp[y_col], x=grp[x_col], text=grp.index, name=str(grp.name), hovertemplate=hover_template)
                traces.append(trace)
            df.groupby(category_col).apply(create_traces)
        else:
            hover_template = '<b>'+x_col+'</b>: %{x}<br><b>' + y_col+'</b>: %{y}' 
            if date_col is not None and date_col in df.columns:
                hover_template += '<br><b>'+date_col+'</b>: %{customdata}'
                trace = go.Scattergl(marker=dict(color=self.controller.PLOT_PALETTE_HEX[0]), mode='markers', y=df[y_col], x=df[x_col], text=df.index, customdata=df[date_col], name=x_col, hovertemplate=hover_template)
            else:
                trace = go.Scattergl(marker=dict(color=self.controller.PLOT_PALETTE_HEX[0]), mode='markers', y=df[y_col], x=df[x_col], text=df.index, name=x_col, hovertemplate=hover_template)
            traces.append(trace)

        if add_trendline:
            slope_df = df[[x_col,y_col]].dropna()
            if slope_df.shape[0] > 1:
                slope, intercept, r_value, p_value, std_err = linregress(slope_df[x_col],slope_df[y_col])

                min_x, max_x = min(slope_df[x_col]), max(slope_df[x_col])
                padding = (max_x - min_x) * 0.1
                min_x -= padding
                max_x += padding
                slope_x = np.array([min_x, max_x])
                slope_y = np.array([intercept + min_x*slope, intercept + max_x*slope])

                line_options = {'color': 'red', 'width': 3, 'shape': 'linear'}
                trend_trace = go.Scattergl(mode='lines', y=slope_y, x=slope_x, line=line_options, name="Cor: " + str(round(r_value*100, 4)) + "% <br>Slope: " + str(round(slope,6)))
                traces.append(trend_trace)

        fig = go.Figure(data=traces, layout=self.layout)
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            font_family=self.controller.PLOT_FONT_FAMILY,
            font_color=self.controller.FONT_COLOR,
            font_size=18
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_layout(hovermode='closest')
        self.show_fig(fig)

    def scatter3d(self, df, x_col, y_col, z_col, date_col=None, category_col=None, add_plane=True):
        # TODO: add plane https://plotly.com/python/3d-surface-plots/
        traces = []
        if category_col is not None and category_col in df.columns:
            def create_traces(grp):
                hover_template = '<b>'+x_col+'</b>: %{x}<br><b>' + z_col+'</b>: %{y}<br><b>' + y_col + '</b>: %{z}<br><b>' + category_col+'</b>: '+str(grp.name)
                if date_col is not None and date_col in grp.columns:
                    hover_template += '<br><b>'+date_col+'</b>: %{customdata}'
                    trace = go.Scatter3d(mode='markers', y=grp[z_col], x=grp[x_col], z=grp[y_col], text=grp.index, customdata=grp[date_col], name=str(grp.name), hovertemplate=hover_template)
                else:
                    trace = go.Scatter3d(mode='markers', y=grp[z_col], x=grp[x_col], z=grp[y_col], text=grp.index, name=str(grp.name), hovertemplate=hover_template)
                traces.append(trace)
            df.groupby(category_col).apply(create_traces)
        else:
            hover_template = '<b>'+x_col+'</b>: %{x}<br><b>' + z_col+'</b>: %{y}<br><b>' + y_col + '</b>: %{z}'
            if date_col is not None and date_col in df.columns:
                hover_template += '<br><b>'+date_col+'</b>: %{customdata}'
                trace = go.Scatter3d(marker=self.uniform_marker_style, mode='markers', y=df[z_col], x=df[x_col], z=df[y_col], text=df.index, customdata=df[date_col], name=x_col, hovertemplate=hover_template)
            else:
                trace = go.Scatter3d(marker=self.uniform_marker_style, mode='markers', y=df[z_col], x=df[x_col], z=df[y_col], text=df.index, name=x_col, hovertemplate=hover_template)
            traces.append(trace)

        if add_plane:
            X = df[[x_col, z_col]]
            y = df[y_col]
            regr = LinearRegression()
            regr.fit(X, y)

            xrange = np.linspace(X[x_col].min(), X[x_col].max())
            zrange = np.linspace(X[z_col].min(), X[z_col].max())
            xx, zz = np.meshgrid(xrange, zrange)
            pred = regr.predict(np.c_[xx.ravel(), zz.ravel()])
            pred = pred.reshape(xx.shape)

            hover_template = '<b>'+x_col+'</b>: %{x}<br><b>' + z_col+'</b>: %{z}<br><b>' + y_col + '</b>: %{y}'
            plane = go.Surface(x=xrange, y=zrange, z=pred, showscale=False, name="Plane", hovertemplate=hover_template)
            traces.append(plane)

        fig = go.Figure(data=traces, layout=self.layout)
        fig.update_layout(scene = dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            font_family=self.controller.PLOT_FONT_FAMILY,
            font_color=self.controller.FONT_COLOR,
            font_size=18
        )
        self.show_fig(fig)

    def line_graph(self, x, y):
        fig = go.Figure(data=[go.Scatter(marker=self.uniform_marker_style, x=x, y=y)], layout=self.layout)
        fig.update_layout(
            font_family=self.controller.PLOT_FONT_FAMILY,
            font_color=self.controller.FONT_COLOR,
            font_size=18
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_layout(hovermode='closest')
        self.show_fig(fig)

    def line_graph_dual_axis(self, x, y1, y2):
        #https://plotly.com/python/multiple-axes/
        fig = go.Figure(layout=self.layout)
        fig.add_trace(go.Scatter(marker=dict(color=self.controller.PLOT_PALETTE_HEX[0]), x=x, y=y1, name=y1.name))
        fig.add_trace(go.Scatter(marker=dict(color=self.controller.PLOT_PALETTE_HEX[1]), x=x, y=y2, name=y2.name, yaxis="y2"))
        fig.update_layout(
            font_family=self.controller.PLOT_FONT_FAMILY,
            font_color=self.controller.FONT_COLOR,
            font_size=18,
            yaxis=dict(
                title=y1.name,
                titlefont=dict(
                    color=self.controller.PLOT_PALETTE_HEX[0]
                ),
                tickfont=dict(
                    color=self.controller.PLOT_PALETTE_HEX[0]
                )
            ),
            yaxis2=dict(
                title=y2.name,
                titlefont=dict(
                    color=self.controller.PLOT_PALETTE_HEX[1]
                ),
                tickfont=dict(
                    color=self.controller.PLOT_PALETTE_HEX[1]
                ),
                anchor="x",
                overlaying="y",
                side="right"
            )
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_layout(hovermode='closest')
        self.show_fig(fig)

    def bar_graph(self, x, y):
        fig = go.Figure(data=[go.Bar(marker=self.uniform_marker_style, x=x, y=y)], layout=self.layout)
        fig.update_layout(
            font_family=self.controller.PLOT_FONT_FAMILY,
            font_color=self.controller.FONT_COLOR,
            font_size=18
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_layout(hovermode='closest')
        self.show_fig(fig)

    def histogram(self, x):
        fig = go.Figure(data=[go.Histogram(marker=self.uniform_marker_style, x=x)], layout=self.layout)
        fig.update_layout(
            font_family=self.controller.PLOT_FONT_FAMILY,
            font_color=self.controller.FONT_COLOR,
            font_size=18
        )
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        fig.update_layout(hovermode='closest')
        self.show_fig(fig)

    def clear_plot(self):
        self.display_html("")
        self.current_selection = []
        self.dates             = []
        self.dates_str         = []

    def show_fig(self, fig):
        #html = pio.to_html(fig, include_plotlyjs=False)
        #self.display_html(html)
        fig.update_layout(template=self.template)
        file_nm = r'pyca\\tmp\\plotly_file_'+self.id+'.html'
        pio.write_html(fig, file_nm, config=self.config)
        self.display_file(file_nm)


class WrapTextItemDelegate(QtWidgets.QAbstractItemDelegate):

    def paint(self, painter, option, index):
        painter.save()
        painter.setBrush(Qt.red)
        painter.setPen(Qt.white)
        painter.setBackground(QBrush(QColor(255,0,0)))
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(15, 250)


class IntInput(QtWidgets.QLineEdit):
    valid_int_changed = pyqtSignal(int)

    def __init__(self, parent=None, bottom=0, top=100, starting_value=0):
        super(IntInput, self).__init__(parent)
        self.bottom         = bottom
        self.top            = top
        self.starting_value = starting_value
        # validator = QtGui.QIntValidator(
        #         bottom, # bottom
        #         top, # top
        #     )
        # self.setValidator(validator)
        self.setText(str(starting_value))
        self.textEdited.connect(self.validate_input)
    
    def validate_input(self, val):
        try:
            int_val = int(val)
            if int_val < self.bottom:
                int_val = self.bottom
            if int_val > self.top:
                int_val = self.top
            self.setText(str(int_val))
            self.valid_int_changed.emit(int_val)
        except:
            self.setText("")

        return


class ExtendedComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None, size_policy=None):
        super(ExtendedComboBox, self).__init__(parent)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable(True)

        if size_policy is not None:
            self.setSizeAdjustPolicy(size_policy)

        #self.setItemDelegate(WrapTextItemDelegate(self))

        # add a filter model to filter matching items
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)


    # on selection of an item from the completer, select the corresponding item from combobox 
    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            self.activated[str].emit(self.itemText(index))


    # on model change, update the models of the filter and completer as well 
    def setModel(self, model):
        super(ExtendedComboBox, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)


    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(ExtendedComboBox, self).setModelColumn(column) 


class ColumnComboBox(ExtendedComboBox):
    def __init__(self, parent, controller, default=""):
        super(ColumnComboBox, self).__init__(parent)
        self.controller = controller
        self.controller.column_lists.append(self)
        if self.controller.is_df_loaded:
            self.addItems([""] + self.controller.all_cols)

            if isinstance(default, str) and default in self.controller.all_cols:
                self.setCurrentIndex(self.controller.all_cols.index(default)+1)

    def update_cols(self, all_cols, x_cols):
        new_cols = [""] + all_cols
        try:
            idx = new_cols.index(self.currentText())
        except ValueError:
            idx = 0
        self.clear()
        self.addItems(new_cols)
        self.setCurrentIndex(idx)

    def deleteLater(self):
        self.controller.column_lists.remove(self)
        super(ColumnComboBox, self).deleteLater()


class XColumnComboBox(ExtendedComboBox):
    def __init__(self, parent, controller, default=""):
        super(XColumnComboBox, self).__init__(parent)
        self.controller = controller
        self.controller.column_lists.append(self)
        if self.controller.is_df_loaded:
            self.addItems([""] + self.controller.x_cols)

            if isinstance(default, str) and default in self.controller.x_cols:
                self.setCurrentIndex(self.controller.x_cols.index(default)+1)

    def update_cols(self, all_cols, x_cols):
        new_cols = [""] + x_cols
        try:
            idx = new_cols.index(self.currentText())
        except ValueError:
            idx = 0
        self.clear()
        self.addItems(new_cols)
        self.setCurrentIndex(idx)

    def deleteLater(self):
        self.controller.column_lists.remove(self)
        super(XColumnComboBox, self).deleteLater()


class SearchableListView(QWidget):
    def __init__(self, parent, controller, title="", wrap=True, multiple=True, track_columns=False, track_x_columns=False):
        super(SearchableListView, self).__init__(parent)
        self.parent              = parent
        self.controller          = controller
        self.track_columns       = track_columns
        self.multiple            = multiple
        self.wrap                = wrap
        self.track_x_columns     = track_x_columns
        self.title               = title

        self.build_layout()

        if track_columns or track_x_columns:
            self.controller.column_lists.append(self)
            if self.controller.is_df_loaded:
                if track_columns:
                    self.setItems(self.controller.all_cols)
                elif track_x_columns:
                    self.setItems(self.controller.x_cols)

    def build_layout(self):
        # Create objects
        self.layout        = QtGui.QVBoxLayout(self)
        self.filter_le     = QtGui.QLineEdit(self)
        self.show_checkbox = QtGui.QCheckBox("Show Selected")
        self.top_line      = QHBoxLayout()
        self.view          = QtGui.QListView(self)
        self.proxy         = QSortFilterProxyModel(self)
        self.model         = QtCore.QStringListModel(self.view)

        self.show_checkbox.stateChanged.connect(self.show_only_selected)

        self.filter_le.textChanged.connect(self.filter_by_le)
        self.filter_le.setPlaceholderText("Filter entries here...")
        if self.multiple:
            self.view.setSelectionMode(QtGui.QListView.MultiSelection)
        self.view.setWordWrap(self.wrap)
        self.view.setFont(self.controller.SETUP_WIDGET_FONT)

        self.proxy.setSourceModel(self.model)
        self.view.setModel(self.proxy)

        if self.title != "" and isinstance(self.title, str):
            self.title_label = QLabel(self.title)
            self.title_label.setFont(self.controller.SETUP_WIDGET_FONT)
            self.layout.addWidget(self.title_label)

        self.top_line.addWidget(self.filter_le)
        self.top_line.addWidget(self.show_checkbox)

        #self.setLayout(self.layout)
        self.layout.addLayout(self.top_line)
        self.layout.addWidget(self.view)

    def setItems(self, items):
        self.model.setStringList(items)

    def selectedItems(self):
        return [row.data() for row in self.view.selectionModel().selectedRows()]

    def update_cols(self, all_cols, x_cols):
        selection_model = self.view.selectionModel()
        # current_selection = self.selectedItems()

        if self.track_columns:
            self.setItems(all_cols)
            self.show_checkbox.setChecked(False)
        elif self.track_x_columns:
            self.setItems(x_cols)
            self.show_checkbox.setChecked(False)

        # for i, string in enumerate(self.model.stringList()):
        #     if string in current_selection:
        #         idx = self.model.index(i, 0)
        #         selection_model.select(idx, QtCore.QItemSelectionModel.ClearAndSelect)

    def show_only_selected(self, show):
        if bool(show):
            current_selection = ["^"+selection+"$" for selection in self.selectedItems()]
            print(current_selection)
            pattern = "|".join(current_selection)
            search = QRegExp(pattern,
                            Qt.CaseInsensitive,
                            QRegExp.RegExp
                            )
            self.proxy.setFilterKeyColumn(0)
            self.proxy.setFilterRegExp(search)
        else:
            self.filter_by_le(self.filter_le.text())

    @QtCore.pyqtSlot(str)
    def filter_by_le(self, text):
        if self.show_checkbox.isChecked():
            return
        
        current_selection = ["^"+selection+"$" for selection in self.selectedItems()] + [text]

        pattern = "|".join(current_selection)
        search = QRegExp(pattern,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
        self.proxy.setFilterKeyColumn(0)
        self.proxy.setFilterRegExp(search)

    def deleteLater(self):
        self.controller.column_lists.remove(self)
        super(SearchableListView, self).deleteLater()


class DownloadQTableView(QtWidgets.QTableView):
    def __init__(self, parent=None, show_row_header=True, cell_font=None, header_font=None, sort_enabled=False):
        super(DownloadQTableView, self).__init__(parent)
        self.installEventFilter(self)
        self.verticalHeader().setVisible(show_row_header)

        self.setSortingEnabled(sort_enabled)

        if cell_font is not None:
            self.setFont(cell_font)
        if header_font is not None:
            self.horizontalHeader().setFont(header_font)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.KeyPress and event.matches(QKeySequence.Copy):
            self.copy_selection()
            return True
        elif event.type() == QtCore.QEvent.KeyPress and event.matches(QKeySequence.Paste):
            self.paste_selection()
            return True
        return super(DownloadQTableView, self).eventFilter(source, event)

    def copy_selection(self):
        selection = self.selectedIndexes()
        if selection:
            all_rows = []
            all_columns = []
            for index in selection:
                if not index.row() in all_rows:
                    all_rows.append(index.row())
                if not index.column() in all_columns:
                    all_columns.append(index.column())
            visible_rows = [row for row in all_rows if not self.isRowHidden(row)]
            visible_columns = [
                col for col in all_columns if not self.isColumnHidden(col)
            ]
            table = [[""] * len(visible_columns) for _ in range(len(visible_rows))]
            for index in selection:
                if index.row() in visible_rows and index.column() in visible_columns:
                    selection_row = visible_rows.index(index.row())
                    selection_column = visible_columns.index(index.column())
                    table[selection_row][selection_column] = index.data()
            stream = io.StringIO()
            csv.writer(stream, delimiter="\t").writerows(table)
            QApplication.clipboard().setText(stream.getvalue())

    def contextMenuEvent(self, event):
        self.menu = QtGui.QMenu(self)
        download_action = QtGui.QAction('Download', self)
        download_action.triggered.connect(lambda: self.download_data(event))
        self.menu.addAction(download_action)
        # add other required actions
        self.menu.popup(QtGui.QCursor.pos())

    def download_data(self, event):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_nm, _ = QFileDialog.getSaveFileName(self,"Save data as CSV", "","CSV files (*.csv)", options=options)
        if file_nm:
            df = self.model().sourceModel().get_df()
            if not file_nm.endswith(".csv"):
                file_nm = file_nm+".csv"
            df.to_csv(file_nm, index=False)


class DefaultWidthQTableView(DownloadQTableView):
    def __init__(self, parent=None, default_width=125, ignore_resize_cols=[0], resize_to_contents_cols=[], *args, **kwargs):
        super(DefaultWidthQTableView, self).__init__(parent, *args, **kwargs)
        self.ignore_resize_cols      = ignore_resize_cols
        self.resize_to_contents_cols = resize_to_contents_cols
        self.default_width           = default_width

    def setModel(self, model):
        super(DefaultWidthQTableView, self).setModel(model)
        self.resizeDefault()
        self.resizeToContents()
    
    def setFont(self, font):
        super(DefaultWidthQTableView, self).setFont(font)
        self.resizeToContents()

    def setHorizontalHeaderFont(self, font):
        self.horizontalHeader().setFont(font)
        self.resizeToContents()

    def resizeToContents(self):
        if self.model() is None:
            return
        for i in range(self.model().columnCount()):
            if i in self.ignore_resize_cols:
                continue

            if i in self.resize_to_contents_cols:
                self.resizeColumnToContents(i)

    def resizeDefault(self):
        if self.model() is None:
            return
        for i in range(self.model().columnCount()):
            if i in self.ignore_resize_cols:
                continue
            self.setColumnWidth(i, self.default_width)


class FlowLayout(QtGui.QLayout):
    # https://stackoverflow.com/questions/41621354/pyqt-wrap-around-layout-of-widgets-inside-a-qscrollarea
    def __init__(self, parent=None, margin=-1, hspacing=-1, vspacing=-1):
        super(FlowLayout, self).__init__(parent)
        self._hspacing = hspacing
        self._vspacing = vspacing
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        del self._items[:]

    def addItem(self, item):
        self._items.append(item)

    def horizontalSpacing(self):
        if self._hspacing >= 0:
            return self._hspacing
        else:
            return self.smartSpacing(
                QtGui.QStyle.PM_LayoutHorizontalSpacing)

    def verticalSpacing(self):
        if self._vspacing >= 0:
            return self._vspacing
        else:
            return self.smartSpacing(
                QtGui.QStyle.PM_LayoutVerticalSpacing)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.doLayout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QtCore.QSize(left + right, top + bottom)
        return size

    def doLayout(self, rect, testonly):
        left, top, right, bottom = self.getContentsMargins()
        effective = rect.adjusted(+left, +top, -right, -bottom)
        x = effective.x()
        y = effective.y()
        lineheight = 0
        for item in self._items:
            widget = item.widget()
            hspace = self.horizontalSpacing()
            if hspace == -1:
                hspace = widget.style().layoutSpacing(
                    QtGui.QSizePolicy.PushButton,
                    QtGui.QSizePolicy.PushButton, QtCore.Qt.Horizontal)
            vspace = self.verticalSpacing()
            if vspace == -1:
                vspace = widget.style().layoutSpacing(
                    QtGui.QSizePolicy.PushButton,
                    QtGui.QSizePolicy.PushButton, QtCore.Qt.Vertical)
            nextX = x + item.sizeHint().width() + hspace
            if nextX - hspace > effective.right() and lineheight > 0:
                x = effective.x()
                y = y + lineheight + vspace
                nextX = x + item.sizeHint().width() + hspace
                lineheight = 0
            if not testonly:
                item.setGeometry(
                    QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint()))
            x = nextX
            lineheight = max(lineheight, item.sizeHint().height())
        return y + lineheight - rect.y() + bottom

    def smartSpacing(self, pm):
        parent = self.parent()
        if parent is None:
            return -1
        elif parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        else:
            return parent.spacing()


class TransformationSummaryBox(QWidget):
    def __init__(self, parent, controller, layout):
        super(TransformationSummaryBox, self).__init__(parent)
        self.controller = controller
        self.parent = parent
        self.parent_layout = layout
        self.build_layout()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignLeft)

        # Top Layout
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        self.tran_label = QLabel("Transformation", self)
        self.tran_label.setAlignment(Qt.AlignCenter)
        self.tran_label.setFont(self.controller.LARGE_FONT)
        self.top_layout.addWidget(self.tran_label)

        # Content Layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setAlignment(Qt.AlignLeft)

        # Add to main layout
        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.content_layout, 10)

        # Create close button
        self.remove_btn = QPushButton("", self)
        self.remove_btn.setIcon(QtGui.QIcon("icons/close.png"))
        self.remove_btn.setIconSize(QtCore.QSize(28,28))
        self.remove_btn.resize(32, 32)
        self.remove_btn.clicked.connect(self.remove_widget)
        self.remove_btn.move(self.width() - self.remove_btn.width() - 1, 1)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(42, 42, 42))
        self.setPalette(p)
    
    def remove_widget(self):
        self.parent.remove_transformation_signal.emit(self.transformation)
        self.parent_layout.removeWidget(self)
        self.deleteLater()
        self = None

    def resizeEvent(self, event):
        # move to top-right corner
        self.remove_btn.move(self.width() - self.remove_btn.width() - 5, 5)
        super(TransformationSummaryBox, self).resizeEvent(event)


class TransformationPopupBox(QWidget):

    def __init__(self, parent, controller):
        super(TransformationPopupBox, self).__init__()
        self.parent = parent
        self.controller = controller
        self.build_layout()        
        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.show()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        
        # Dropdown
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        
        self.dropdown = QtWidgets.QComboBox(self)
        self.dropdown.addItems([""] + list(transformations.transformation_types.keys()))
        self.dropdown.currentTextChanged.connect(self.load_transformation)
        self.top_layout.addWidget(self.dropdown)

        # Content
        # self.content_layout = QtWidgets.QVBoxLayout()
        # self.scrollArea = QtWidgets.QScrollArea(self)
        # self.scrollArea.setWidgetResizable(True)
        # self.scrollAreaWidgetContents = QtWidgets.QWidget(self)
        # self.widget_layout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        # self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        # self.content_layout.addWidget(self.scrollArea)
        self.widget_layout = QtWidgets.QVBoxLayout()
        self.widget_layout.setAlignment(Qt.AlignTop)
        
        # Confirm
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_layout.setAlignment(Qt.AlignCenter)
        self.confirm_btn = QPushButton("Create Transformation", self)
        self.confirm_btn.clicked.connect(self.validate_inputs)
        self.confirm_btn.setFont(self.controller.NORMAL_BTN_FONT)
        self.confirm_btn.setFixedSize(225, 50)
        self.btn_layout.addWidget(self.confirm_btn)

        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.widget_layout, 10)
        self.main_layout.addLayout(self.btn_layout, 1)

    def validate_inputs(self):
        self.controller.start_status_task("Creating " + self.transformation.name + " transformation.")
        if self.transformation.validate_inputs(self):
            self.parent.create_transformation_signal.emit(self.transformation)
            self.unload_transformation()
            self.close()
        else:
            self.controller.finished_status_task()

    def load_transformation(self, value):
        self.unload_transformation()
        try:
            self.transformation = transformations.transformation_types[self.dropdown.currentText()](self.controller)
        except KeyError:
            return
        self.transformation.add_widgets(self.widget_layout, self, self.controller)

    def unload_transformation(self):
        self.transformation = None
        self.clear_layout(self.widget_layout)

    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            try:
                item = layout.takeAt(i)
                if item.layout() is not None:
                    self.clear_layout(item.layout())
                elif item.widget() is not None:
                    item.widget().deleteLater()
                    layout.removeWidget(item.widget())
            except Exception as e:
                print(str(e))


    def closeEvent(self, event):
        self.unload_transformation()
        event.accept()


class FilterSummaryBox(QWidget):
    def __init__(self, parent, controller, layout):
        super(FilterSummaryBox, self).__init__(parent)
        self.controller = controller
        self.parent = parent
        self.parent_layout = layout
        self.build_layout()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignLeft)

        # Top Layout
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        self.filter_label = QLabel("Filter", self)
        self.filter_label.setAlignment(Qt.AlignCenter)
        self.filter_label.setFont(self.controller.LARGE_FONT)
        self.top_layout.addWidget(self.filter_label)

        # Content Layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setAlignment(Qt.AlignLeft)

        # Add to main layout
        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.content_layout, 10)

        # Create close button
        self.remove_btn = QPushButton("", self)
        self.remove_btn.setIcon(QtGui.QIcon("icons/close.png"))
        self.remove_btn.setIconSize(QtCore.QSize(28,28))
        self.remove_btn.resize(32, 32)
        self.remove_btn.clicked.connect(self.remove_widget)
        self.remove_btn.move(self.width() - self.remove_btn.width() - 1, 1)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtGui.QColor(42, 42, 42))
        self.setPalette(p)
    
    def remove_widget(self):
        self.parent.remove_filter_signal.emit(self.filter)
        self.parent_layout.removeWidget(self)
        self.deleteLater()
        self = None

    def resizeEvent(self, event):
        # move to top-right corner
        self.remove_btn.move(self.width() - self.remove_btn.width() - 5, 5)
        super(FilterSummaryBox, self).resizeEvent(event)


class FilterPopupBox(QWidget):

    def __init__(self, parent, controller):
        super(FilterPopupBox, self).__init__()
        self.parent = parent
        self.controller = controller
        self.build_layout()        
        self.setMinimumHeight(500)
        self.setMinimumWidth(500)
        self.show()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        
        # Dropdown
        self.top_layout = QHBoxLayout()
        self.top_layout.setAlignment(Qt.AlignCenter)
        
        self.dropdown = QtWidgets.QComboBox(self)
        self.dropdown.addItems([""] + list(filters.filter_types.keys()))
        self.dropdown.currentTextChanged.connect(self.load_filter)
        self.top_layout.addWidget(self.dropdown)

        # Content
        # self.content_layout = QtWidgets.QVBoxLayout()
        # self.scrollArea = QtWidgets.QScrollArea(self)
        # self.scrollArea.setWidgetResizable(True)
        # self.scrollAreaWidgetContents = QtWidgets.QWidget(self)
        # self.widget_layout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        # self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        # self.content_layout.addWidget(self.scrollArea)
        self.widget_layout = QtWidgets.QVBoxLayout()
        self.widget_layout.setAlignment(Qt.AlignTop)
        
        # Confirm
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_layout.setAlignment(Qt.AlignCenter)
        self.confirm_btn = QPushButton("Apply Filter", self)
        self.confirm_btn.clicked.connect(self.validate_inputs)
        self.confirm_btn.setFont(self.controller.NORMAL_BTN_FONT)
        self.confirm_btn.setFixedSize(225, 50)
        self.btn_layout.addWidget(self.confirm_btn)

        self.main_layout.addLayout(self.top_layout, 1)
        self.main_layout.addLayout(self.widget_layout, 10)
        self.main_layout.addLayout(self.btn_layout, 1)

    def validate_inputs(self):
        self.controller.start_status_task("Creating " + self.filter.name + " filter.")
        if self.filter.validate_inputs(self):
            self.parent.create_filter_signal.emit(self.filter)
            self.unload_filter()
            self.close()
        else:
            self.controller.finished_status_task()

    def load_filter(self, value):
        self.unload_filter()
        try:
            self.filter = filters.filter_types[self.dropdown.currentText()]()
        except KeyError:
            return
        self.filter.add_widgets(self.widget_layout, self, self.controller)

    def unload_filter(self):
        self.filter = None
        for i in reversed(range(self.widget_layout.count())): 
            widgetToRemove = self.widget_layout.itemAt(i).widget()
            self.widget_layout.removeWidget(widgetToRemove)
            widgetToRemove.deleteLater()

    def closeEvent(self, event):
        self.unload_filter()
        event.accept()


class FilterScatterWidget(QWidget):

    metric_dive_filter_signal = pyqtSignal(object)
    remove_metric_dive_filter_signal = pyqtSignal()
    dates_updated_signal = pyqtSignal(list, list)

    def __init__(self, parent=None, controller=None):
        super(FilterScatterWidget, self).__init__()

        self.dates     = []
        self.dates_str = []

        self.parent = parent
        self.controller = controller

        self.metric_dive_filter_signal.connect(self.controller.dataWorker.create_metric_dive_filter)
        self.remove_metric_dive_filter_signal.connect(self.controller.dataWorker.remove_metric_dive_filters)

        self.build_layout()

    def build_layout(self):
        self.main_layout = QHBoxLayout(self)
        self.scatter = PlotlyView(self.controller)
        self.dates_updated_signal.connect(self.update_date_box)

        # Filter Side
        self.filter_side = QVBoxLayout()
        self.filter_side.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.filter_side_holder = QWidget(self)
        self.filter_side_holder.setLayout(self.filter_side)
        self.filter_side_holder.setMaximumWidth(350)
        self.filter_check = QCheckBox("Filter", self)
        self.filter_check.stateChanged.connect(self.toggle_filter)
        self.filter_check.setFont(self.controller.NORMAL_FONT)
        self.filter_side.addWidget(self.filter_check)

        self.main_layout.addWidget(self.filter_side_holder)
        self.main_layout.addWidget(self.scatter)

    def enable_actions(self, enable):
        self.filter_check.setEnabled(enable)

        objs = [self.keep_btn, self.filter_check, self.remove_btn, self.date_keep_btn, self.date_remove_btn]
        for obj in objs:
            try:
                obj.setEnabled(enable)
            except:
                pass

    def toggle_filter(self):
        if self.filter_check.isChecked():
            self.build_filter()
        else:
            self.remove_filter()

    def enable_filter(self, enable):
        self.filter_check.setChecked(enable)

        if enable:
            self.build_filter()
        else:
            self.remove_filter()

    def build_filter(self):
        self.filter_contents = QVBoxLayout()

        self.filter_by_scatter_label = QLabel("Filter By Scatter", self)
        self.filter_by_scatter_label.setFont(self.controller.NORMAL_FONT)
        self.button_layout = QHBoxLayout()
        self.keep_btn = QPushButton("Keep Selection")
        self.keep_btn.clicked.connect(lambda: self.apply_scatter_filter(True))
        self.remove_btn = QPushButton("Remove Selection")
        self.remove_btn.clicked.connect(lambda: self.apply_scatter_filter(False))
        self.button_layout.addWidget(self.keep_btn)
        self.button_layout.addWidget(self.remove_btn)
        
        self.filter_by_date_label = QLabel("Filter By Date", self)
        self.filter_by_date_label.setFont(self.controller.NORMAL_FONT)
        self.date_box = QListWidget(self)
        self.date_box.setSelectionMode(QListWidget.MultiSelection)
        self.update_date_box(self.dates, self.dates_str)

        self.date_button_layout = QHBoxLayout()
        self.date_keep_btn = QPushButton("Keep Selected Dates")
        self.date_keep_btn.clicked.connect(lambda: self.apply_date_filter(True))
        self.date_remove_btn = QPushButton("Remove Selected Dates")
        self.date_remove_btn.clicked.connect(lambda: self.apply_date_filter(False))
        self.date_button_layout.addWidget(self.date_keep_btn)
        self.date_button_layout.addWidget(self.date_remove_btn)

        self.filter_contents.addWidget(self.filter_by_scatter_label)
        self.filter_contents.addLayout(self.button_layout)
        self.filter_contents.addWidget(self.filter_by_date_label)
        self.filter_contents.addWidget(self.date_box)
        self.filter_contents.addLayout(self.date_button_layout)

        self.filter_side.addLayout(self.filter_contents)

    def remove_filter(self):
        if hasattr(self, "filter_side") and hasattr(self, "filter_contents"):
            self.filter_side.removeItem(self.filter_contents)
        
        if hasattr(self, "filter_by_scatter_label"):
            self.filter_by_scatter_label.deleteLater()

        if hasattr(self, "button_layout"):
            self.button_layout.deleteLater()

        if hasattr(self, "keep_btn"):
            self.keep_btn.deleteLater()

        if hasattr(self, "remove_btn"):
            self.remove_btn.deleteLater()

        if hasattr(self, "filter_by_date_label"):
            self.filter_by_date_label.deleteLater()

        if hasattr(self, "date_box"):
            self.date_box.deleteLater()

        if hasattr(self, "date_button_layout"):
            self.date_button_layout.deleteLater()

        if hasattr(self, "date_keep_btn"):
            self.date_keep_btn.deleteLater()

        if hasattr(self, "date_remove_btn"):
            self.date_remove_btn.deleteLater()

        self.remove_metric_dive_filter_signal.emit()

    def set_dates(self, df, date_col=None):
        if date_col is not None and df is not None and date_col in df.columns:
            try:
                self.dates     = list(df[date_col].sort_values().unique())
                self.dates_str = list(pd.to_datetime(self.dates).strftime("%Y-%m-%d"))
            except Exception as e:
                traceback.print_exc()
                print(str(e) + " on 'set_dates' in widgets.py")
                self.dates     = []
                self.dates_str = []
        else:
            self.dates     = []
            self.dates_str = []

        self.dates_updated_signal.emit(self.dates, self.dates_str)

    def update_date_box(self, dates, dates_str):
        if hasattr(self, "date_box"):
            try:
                self.date_box.clear()
                self.date_box.addItems(self.dates_str)
                # Select all dates initiially
                for date in self.dates_str:
                    matching_items = self.date_box.findItems(date, Qt.MatchExactly)
                    for item in matching_items:
                        item.setSelected(True)
            except RuntimeError:
                pass

    def apply_date_filter(self, keep=False):
        filter_dates_idx = [self.dates_str.index(self.date_box.item(i).text()) for i in range(self.date_box.count()) if self.date_box.item(i).isSelected()]
        filter_dates = [self.dates[idx] for idx in filter_dates_idx]
        if len(filter_dates) == 0:
            return
        date_filter = filters.MetricDiveDateFilter(filter_dates, keep)
        self.parent.clear_all(False)
        self.metric_dive_filter_signal.emit(date_filter)

    def apply_scatter_filter(self, keep=False):
        selected_points = self.scatter.get_selected_points()
        if len(selected_points) == 0:
            return
        point_filter = filters.MetricDivePlotlyPointFilter(selected_points, keep)
        self.parent.clear_all(False)
        self.metric_dive_filter_signal.emit(point_filter)


class QuickFilterTable(QGridLayout):
    def __init__(self, parent, controller, filter_col, *args, **kwargs):
        super(QuickFilterTable, self).__init__(parent)
        self.parent     = parent
        self.controller = controller

        if not isinstance(filter_col, int):
            raise ValueError('filter_col must be an integer corresponding to the column index that should be filtered.')

        self.filter_col = filter_col

        self.line_edit   = QLineEdit(self.parent)
        self.table_view  = DefaultWidthQTableView(self.parent, *args, **kwargs)
        label            = QLabel("Filter", self.parent)

        # Set up grid
        self.addWidget(label, 0, 0, 1, 1)
        self.addWidget(self.line_edit, 0, 1, 1, 1)
        self.addWidget(self.table_view, 1, 0, 1, 4)

        # Set object options
        self.line_edit.textChanged.connect(self.filter_table)

    def buildModel(self, model):
        self.proxy = NumericSortProxyModel(self)
        self.proxy.setSourceModel(model)
        self.table_view.setModel(self.proxy)

    @QtCore.pyqtSlot(str)
    def filter_table(self, text):
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
        self.proxy.setFilterKeyColumn(self.filter_col)
        self.proxy.setFilterRegExp(search)


class ColumnFilterTable(QGridLayout):
    def __init__(self, parent, controller, numeric_sort=True, *args, **kwargs):
        super(ColumnFilterTable, self).__init__(parent)
        self.parent       = parent
        self.controller   = controller
        self.numeric_sort = numeric_sort

        self.lineEdit   = QLineEdit(self.parent)
        self.tableView  = DownloadQTableView(self.parent, show_row_header=True, *args, **kwargs)
        self.comboBox   = ExtendedComboBox(self.parent)
        self.label      = QLabel(self.parent)
        self.btn        = QPushButton("Apply Filter", self.parent)

        self.addWidget(self.label,     0, 0, 1, 1)
        self.addWidget(self.lineEdit,  0, 1, 1, 15)
        self.addWidget(self.comboBox,  1, 0, 1, 15)
        self.addWidget(self.btn,       1, 15, 1, 1)
        self.addWidget(self.tableView, 2, 0, 1, 16)

        self.label.setText("Filter")
        self.btn.clicked.connect(self.search_model)
        
        self.tableView.setSortingEnabled(True)
    
    def buildModel(self, model):
        if self.numeric_sort:
            self.proxy = NumericSortProxyModel(self)
        else:
            self.proxy = AccurateRowNumberProxy(self)
        self.proxy.setSourceModel(model)
        self.tableView.setModel(self.proxy)
        self.set_columns(model.get_cols())

    def set_columns(self, cols):
        new_cols = [""] + list(cols)
        self.comboBox.clear()
        self.comboBox.addItems(new_cols)   
 
    def search_model(self):
        text = self.lineEdit.text()
        index = self.comboBox.currentIndex()-1
        search = QRegExp(text,
                         Qt.CaseInsensitive,
                         QRegExp.RegExp
                         )
                         
        self.proxy.setFilterKeyColumn(index)
        self.proxy.setFilterRegExp(search)



class PrettyTable(QWidget):
    def __init__(self, n_rows, n_columns, parent=None, controller=None, title=None, column_headers=None):
        super(PrettyTable, self).__init__(parent)
        self.parent         = parent
        self.controller     = controller
        self.n_rows         = n_rows
        self.n_columns      = n_columns
        self.title          = title
        self.column_headers = column_headers
        self.table          = [[None]*n_columns for c in range(n_rows)]
        self.header_labels  = [None]*n_columns

        self.bold_fnt = QFont(self.controller.FONT_BASE, 16)
        self.bold_fnt.setBold(True)

        self.build_grid()

    def build_grid(self):
        self.layout = QGridLayout(self)
        self.layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        if isinstance(self.title, str):
            self.title_label = QLabel(self.title, self)
            title_font = QFont(self.controller.FONT_BASE, 22)
            title_font.setUnderline(True)
            self.title_label.setFont(title_font)
            self.title_label.setAlignment(Qt.AlignCenter) 
            self.layout.addWidget(self.title_label,0,0,1,self.n_columns)

        if isinstance(self.column_headers, list) and all([isinstance(header, str) for header in self.column_headers]):
            if len(self.column_headers) != self.n_columns:
                raise Exception("column_headers needs to be a list of length equal to n_columns.")

            for i, header in enumerate(self.column_headers):
                label = QLabel(header, self)
                label.setFont(self.bold_fnt)
                label.setAlignment(Qt.AlignCenter)
                self.header_labels[i] = label 
                self.layout.addWidget(label, 1, i)

        for r in range(self.n_rows):
            for c in range(self.n_columns):
                cell = QLabel("--", self)
                cell.setFont(self.controller.SMALL_FONT)
                cell.setAlignment(Qt.AlignCenter)
                self.layout.addWidget(cell, r+2, c)
                self.table[r][c] = cell

    def set_column_values(self, column, values):
        if not isinstance(values, list) or len(values) != self.n_rows or not all([isinstance(value, str) for value in values]):
            raise Exception("values must be a list of str of length equal to n_rows.")

        for i, value in enumerate(values):
            self.table[i][column].setText(value)


class Loader(QWidget):
    def __init__(self, content):
        super(Loader, self).__init__()

        self.stack  = QStackedWidget()
        self.stack.addWidget(QWidget())
        self.stack.addWidget(QWidget())
        # self.grid   = QGridLayout(self)

        # self.grid.addWidget(self.stack, 0, 0, 1, 1, Qt.AlignCenter)
        # self.grid.setColumnStretch(0, 1)
        # self.grid.setRowStretch(0, 1)

        self.stack.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)

        #self.loader  = QLabel("Loading...")

        #self.content = content

        #self.stack.addWidget(self.loader)
        #self.stack.addWidget(self.content)

        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet('background-color: red')

        self.stack.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.stack.setStyleSheet('background-color: green')

        #self.stack.setCurrentIndex(0)

    def start_load(self):
        self.stack.setCurrentIndex(0)
        return
    
    def finish_load(self):
        self.stack.setCurrentIndex(1)
        return



class DetailsTab(QWidget):
    def __init__(self, parent=None, controller=None):
        super(DetailsTab, self).__init__()
        self.parent = parent
        self.controller = controller

        self.controller.dataWorker.metric_dive_perf_date_table_signal.connect(self.build_metric_dive_performance_date_proxy)
        self.controller.dataWorker.metric_dive_perf_category_table_signal.connect(self.build_metric_dive_performance_category_proxy)

        #self.setAutoFillBackground(True)
        #p = self.palette()
        #p.setColor(self.backgroundRole(), Qt.red)
        #self.setPalette(p)

        self.build_layout()

    def build_layout(self):
        self.main_layout = QVBoxLayout(self)
        self.top_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()

        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.bottom_layout)

        bold_fnt = QFont(self.controller.FONT_BASE, 16)
        bold_fnt.setBold(True)

        #
        # ANOVA Section
        #
        self.anova_layout = QVBoxLayout()
        self.anova_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        lin_sum_label = QLabel("   ANOVA   ", self)
        anova_font = QFont(self.controller.FONT_BASE, 28)
        anova_font.setUnderline(True)
        lin_sum_label.setFont(anova_font) 
        self.anova_layout.addWidget(lin_sum_label)

        # Data Points
        dp_label = QLabel("Data Points", self)
        dp_label.setFont(bold_fnt) 
        dp_label.setAlignment(Qt.AlignCenter) 
        self.anova_layout.addWidget(dp_label)

        self.dp = QLabel("--", self)
        self.dp.setFont(self.controller.SMALL_FONT)
        self.dp.setAlignment(Qt.AlignCenter) 
        self.anova_layout.addWidget(self.dp)

        # R-Squared
        r_squared_label = QLabel("R-Squared", self)
        r_squared_label.setFont(bold_fnt)
        r_squared_label.setAlignment(Qt.AlignCenter)  
        self.anova_layout.addWidget(r_squared_label)

        self.r_squared = QLabel("--", self)
        self.r_squared.setFont(self.controller.SMALL_FONT) 
        self.r_squared.setAlignment(Qt.AlignCenter) 
        self.anova_layout.addWidget(self.r_squared)

        # Correlation
        correlation_label = QLabel("Correlation", self)
        correlation_label.setFont(bold_fnt) 
        correlation_label.setAlignment(Qt.AlignCenter)
        self.anova_layout.addWidget(correlation_label)

        self.correlation = QLabel("--", self)
        self.correlation.setFont(self.controller.SMALL_FONT) 
        self.correlation.setAlignment(Qt.AlignCenter)
        self.anova_layout.addWidget(self.correlation)

        # P-Value
        p_value_label = QLabel("P-Value", self)
        p_value_label.setFont(bold_fnt) 
        p_value_label.setAlignment(Qt.AlignCenter)
        self.anova_layout.addWidget(p_value_label)

        self.p_value = QLabel("--", self)
        self.p_value.setFont(self.controller.SMALL_FONT) 
        self.p_value.setAlignment(Qt.AlignCenter)
        self.anova_layout.addWidget(self.p_value)

        # Slope
        slope_label = QLabel("Slope", self)
        slope_label.setFont(bold_fnt) 
        slope_label.setAlignment(Qt.AlignCenter)
        self.anova_layout.addWidget(slope_label)

        self.slope = QLabel("--", self)
        self.slope.setFont(self.controller.SMALL_FONT)
        self.slope.setAlignment(Qt.AlignCenter) 
        self.anova_layout.addWidget(self.slope)

        # ~~~~~~~~~~~~~~~ #
        # SUMMARY Section #
        # ~~~~~~~~~~~~~~~ #

        # Create layout and label
        self.summary_layout = QGridLayout()
        self.summary_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        summary_label = QLabel("   Summary   ", self)
        summary_font = QFont(self.controller.FONT_BASE, 28)
        summary_font.setUnderline(True)
        summary_label.setFont(summary_font)
        summary_label.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(summary_label,0,0,1,3)

        # Column headings #
        metric_label = QLabel("Metric", self)
        metric_label.setFont(bold_fnt)
        metric_label.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(metric_label, 1, 0)

        self.x_label = QLabel("X", self)
        self.x_label.setFont(bold_fnt)
        self.x_label.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.x_label, 1, 1)

        self.y_label = QLabel("Y", self)
        self.y_label.setFont(bold_fnt)
        self.y_label.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.y_label, 1, 2)

        # Metric Labels #
        # Data Points
        summary_dp_label = QLabel("Data Points", self)
        summary_dp_label.setFont(self.controller.SMALL_FONT)
        self.summary_layout.addWidget(summary_dp_label, 2, 0)

        self.x_dp_value = QLabel("--", self)
        self.x_dp_value.setFont(self.controller.SMALL_FONT)
        self.x_dp_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.x_dp_value, 2, 1)

        self.y_dp_value = QLabel("--", self)
        self.y_dp_value.setFont(self.controller.SMALL_FONT)
        self.y_dp_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.y_dp_value, 2, 2)

        # Standard Deviation
        summary_std_label = QLabel("Standard Deviation", self)
        summary_std_label.setFont(self.controller.SMALL_FONT)
        self.summary_layout.addWidget(summary_std_label, 3, 0)

        self.x_std_value = QLabel("--", self)
        self.x_std_value.setFont(self.controller.SMALL_FONT)
        self.x_std_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.x_std_value, 3, 1)

        self.y_std_value = QLabel("--", self)
        self.y_std_value.setFont(self.controller.SMALL_FONT)
        self.y_std_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.y_std_value, 3, 2)

        # Mean
        summary_mean_label = QLabel("Mean", self)
        summary_mean_label.setFont(self.controller.SMALL_FONT)
        self.summary_layout.addWidget(summary_mean_label, 4, 0)

        self.x_mean_value = QLabel("--", self)
        self.x_mean_value.setFont(self.controller.SMALL_FONT)
        self.x_mean_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.x_mean_value, 4, 1)

        self.y_mean_value = QLabel("--", self)
        self.y_mean_value.setFont(self.controller.SMALL_FONT)
        self.y_mean_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.y_mean_value, 4, 2)

        # Median
        summary_median_label = QLabel("Median", self)
        summary_median_label.setFont(self.controller.SMALL_FONT)
        self.summary_layout.addWidget(summary_median_label, 5, 0)

        self.x_median_value = QLabel("--", self)
        self.x_median_value.setFont(self.controller.SMALL_FONT)
        self.x_median_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.x_median_value, 5, 1)

        self.y_median_value = QLabel("--", self)
        self.y_median_value.setFont(self.controller.SMALL_FONT)
        self.y_median_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.y_median_value, 5, 2)

        # Max
        summary_max_label = QLabel("Max", self)
        summary_max_label.setFont(self.controller.SMALL_FONT)
        self.summary_layout.addWidget(summary_max_label, 6, 0)

        self.x_max_value = QLabel("--", self)
        self.x_max_value.setFont(self.controller.SMALL_FONT)
        self.x_max_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.x_max_value, 6, 1)

        self.y_max_value = QLabel("--", self)
        self.y_max_value.setFont(self.controller.SMALL_FONT)
        self.y_max_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.y_max_value, 6, 2)

        # Min
        summary_min_label = QLabel("Min", self)
        summary_min_label.setFont(self.controller.SMALL_FONT)
        self.summary_layout.addWidget(summary_min_label, 7, 0)

        self.x_min_value = QLabel("--", self)
        self.x_min_value.setFont(self.controller.SMALL_FONT)
        self.x_min_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.x_min_value, 7, 1)

        self.y_min_value = QLabel("--", self)
        self.y_min_value.setFont(self.controller.SMALL_FONT)
        self.y_min_value.setAlignment(Qt.AlignCenter) 
        self.summary_layout.addWidget(self.y_min_value, 7, 2)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~ #
        # PERFORMANCE TABLE SECTION #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~ #

        self.perf_tabs = QTabWidget(self)

        self.perf_date_view = DefaultWidthQTableView(self, cell_font=self.controller.TABLE_FONT_MEDIUM, header_font=self.controller.TABLE_FONT_LARGE)
        self.perf_tabs.addTab(self.perf_date_view, "By Date")

        self.perf_category_view = DefaultWidthQTableView(self, cell_font=self.controller.TABLE_FONT_MEDIUM, header_font=self.controller.TABLE_FONT_LARGE)
        self.perf_tabs.addTab(self.perf_category_view, "By Category")

        self.bottom_layout.addWidget(self.perf_tabs)

        # Add sub-layouts to main layout #
        self.top_layout.addLayout(self.anova_layout, 1)
        self.top_layout.addLayout(self.summary_layout, 1)

    def set_anova_values(self, stats_dict):
        if stats_dict == None:
            stats_dict = {
                'dp': '--',
                'r_squared': '--',
                'correlation': '--',
                'p_value': '--',
                'slope': '--'
            }

        self.dp.setText(str(round_default(stats_dict['dp'], 0)))
        self.r_squared.setText(str(round_default(stats_dict['r_squared']*100, 4)) +"%")
        self.correlation.setText(str(round_default(stats_dict['correlation']*100, 4)) +"%") 
        self.p_value.setText(str(round_default(stats_dict['p_value']*100, 4)) +"%")
        self.slope.setText(str(round_default(stats_dict['slope'],12)))

    def set_summary_values(self, stats_dict):
        if stats_dict == None:
            stats_dict = {
                'x_name': '--',
                'x_dp': '--',
                'x_std': '--',
                'x_mean': '--',
                'x_median': '--',
                'x_max': '--',
                'x_min': '--',
                'y_name': '--',
                'y_dp': '--',
                'y_std': '--',
                'y_mean': '--',
                'y_median': '--',
                'y_max': '--',
                'y_min': '--'
            }

        # X
        self.x_label.setText(stats_dict['x_name'])
        self.x_dp_value.setText(str(round_default(stats_dict['x_dp'], 0)))
        self.x_std_value.setText(str(round_default(stats_dict['x_std'], 4)))
        self.x_mean_value.setText(str(round_default(stats_dict['x_mean'], 4)))
        self.x_median_value.setText(str(round_default(stats_dict['x_median'], 4)))
        self.x_max_value.setText(str(round_default(stats_dict['x_max'], 4)))
        self.x_min_value.setText(str(round_default(stats_dict['x_min'], 4)))

        # X
        self.y_label.setText(stats_dict['y_name'])
        self.y_dp_value.setText(str(round_default(stats_dict['y_dp'], 0)))
        self.y_std_value.setText(str(round_default(stats_dict['y_std'], 4)))
        self.y_mean_value.setText(str(round_default(stats_dict['y_mean'], 4)))
        self.y_median_value.setText(str(round_default(stats_dict['y_median'], 4)))
        self.y_max_value.setText(str(round_default(stats_dict['y_max'], 4)))
        self.y_min_value.setText(str(round_default(stats_dict['y_min'], 4)))

    @pyqtSlot(object)
    def build_metric_dive_performance_date_proxy(self, model):
        self.perf_table_proxy = NumericSortProxyModel(self)
        self.perf_table_proxy.setSourceModel(model)
        self.perf_table_proxy.setFilterKeyColumn(0)

        self.perf_date_view.setModel(self.perf_table_proxy)

    @pyqtSlot(object)
    def build_metric_dive_performance_category_proxy(self, model):
        self.perf_category_table_proxy = NumericSortProxyModel(self)
        self.perf_category_table_proxy.setSourceModel(model)
        self.perf_category_table_proxy.setFilterKeyColumn(0)

        self.perf_category_view.setModel(self.perf_category_table_proxy)


class ScatterSelectionViewBox(pg.ViewBox):

    items_selected_signal = pyqtSignal(list)

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            pg.ViewBox.mouseDragEvent(self, ev)
        else:
            ev.accept()

        if ev.button() == QtCore.Qt.LeftButton:
            if ev.isFinish():
                self.rbScaleBox.hide()
                self.selection_box = self.create_rect(ev.buttonDownScenePos(ev.button()), ev.scenePos())
                selection = self.get_items_in_selection()
                if len(selection) > 0:
                    self.items_selected_signal.emit(selection)
            else:
                rect = self.create_rect(ev.buttonDownPos(ev.button()), ev.pos())
                self.updateScaleBox(rect.topLeft(), rect.bottomRight())

    def create_rect(self, p1, p2):
        min_x = min(p1.x(), p2.x())
        min_y = min(p1.y(), p2.y())

        max_x = max(p1.x(), p2.x())
        max_y = max(p1.y(), p2.y())

        return QtCore.QRectF(pg.Point(min_x, min_y), pg.Point(max_x, max_y) )

    def get_items_in_selection(self):
        scatters = [item for item in self.scene().items(self.selection_box) if isinstance(item, pg.graphicsItems.ScatterPlotItem.ScatterPlotItem)]
        if len(scatters) == 0:
            return []

        data_bounds = self.translate_to_plot_coords(self.selection_box, scatters[0])

        selection = []
        for scatter in scatters:
            for point in scatter.data:
                if self.is_point_in_bounds(point, data_bounds):
                    selection.append(point[7])
        return selection

    def translate_to_plot_coords(self, rect, scatter):
        min_x = min(rect.topLeft().x(), rect.bottomRight().x())
        min_y = min(rect.topLeft().y(), rect.bottomRight().y())

        max_x = max(rect.topLeft().x(), rect.bottomRight().x())
        max_y = max(rect.topLeft().y(), rect.bottomRight().y())

        top_left  = pg.Point(min_x, min_y)
        bot_right = pg.Point(max_x, max_y) 
        
        return (QtCore.QRectF(scatter.mapFromScene(top_left), scatter.mapFromScene(bot_right)))

    def is_point_in_bounds(self, point, bounds):
        within_x = bounds.topLeft().x()    <= point[0] <= bounds.topRight().x()
        within_y = bounds.bottomLeft().y() <= point[1] <= bounds.topLeft().y()
        return within_x and within_y 


class ScatterPlot(QObject):

    def __init__(self, parent=None, controller=None, tooltip=False, legend=True, selectable=True, hoverable=True):
        super(ScatterPlot, self).__init__(parent)
        self.parent            = parent
        self.controller        = controller
        self.selectable        = selectable
        self.last_clicked      = []
        self.last_hovered      = []
        self.current_selection = []
        self.series            = {}
        self.x_col             = None
        self.y_col             = None
        self.category_col      = None
        self.ignore_hover      = []

        self.view = pg.GraphicsLayoutWidget()

        if hoverable:
            self.view.sceneObj.sigMouseMoved.connect(self.on_move)

        if selectable:
            self.vb = ScatterSelectionViewBox()
            self.plot = self.view.addPlot(viewBox=self.vb)
            self.vb.items_selected_signal.connect(self.on_items_selected)
        else:
            self.plot = self.view.addPlot()

        if tooltip:
            self.tooltip = pg.TextItem(text="", fill=self.controller.BACKGROUND_SECONDARY, border=self.controller.BACKGROUND_PRIMARY, color=self.controller.FONT_COLOR,anchor=(1,1))
            self.plot.addItem(self.tooltip)
            self.tooltip.setVisible(False)

        if legend:
            self.legend = self.plot.addLegend()
            self.legend.setBrush(pg.mkBrush(self.controller.BACKGROUND_SECONDARY))
            self.legend.setVisible(False)

    def get_selected_coords(self):
        coords = [Coord((point._data[0], point._data[1])) for point in self.current_selection]
        return coords

    def clear_plot(self):
        self.plot.clear()
        if hasattr(self, 'trendline') and self.trendline is not None:
            self.plot.removeItem(self.trendline)
        if hasattr(self, 'legend') and self.legend is not None:
            self.legend.clear()
        self.last_clicked      = []
        self.last_hovered      = []
        self.current_selection = []
        self.series            = {}
        self.x_col             = None
        self.y_col             = None
        self.category_col      = None

    def set_variables(self, df, x, y, category=None):
        self.x_col        = x
        self.y_col        = y
        self.category_col = category

    def set_options(self, category):
        if hasattr(self, 'legend'):
            if category is not None and category != "":
                self.legend.setVisible(True)
            else:
                self.legend.setVisible(False)

        if hasattr(self, 'tooltip'):
            self.tooltip.setVisible(False)

    def quadrant(self, df, x, y, alpha=0.5, category=None, category_color_dict=None, category_size_dict=None):

        self.clear_plot()
        self.set_options(category)
        self.set_variables(df, x, y, category)

        MIN_SIZE = 10
        MAX_SIZE = 300

        if category is not None and category != "" and category in df.columns:
            for i, c in enumerate(df[category].unique()):
                tempdf = df[df[category] == c]

                # Get color from palette
                if category_color_dict is not None and isinstance(category_color_dict, dict):
                    p = category_color_dict.get(c, (255,255,255))
                else:
                    p = self.controller.PLOT_PALETTE_RGB[i].get_tuple()

                # Get size from dict, if available
                if category_size_dict is not None and isinstance(category_size_dict, dict):
                    size = category_size_dict.get(c, 1) * MAX_SIZE

                size = max(MIN_SIZE, size)
                size = min(MAX_SIZE, size)

                series = pg.ScatterPlotItem(size=size, pen=pg.mkPen(None), brush=pg.mkBrush(p[0], p[1], p[2], 255*alpha))
                data_list = [dict(dict(row), **{'color': p}) for index, row in tempdf.iterrows()]
                series.setData(tempdf[x], tempdf[y], data=data_list)
                series.sigClicked.connect(self.on_click)
                self.series[c] = series
                self.plot.addItem(series)
                if hasattr(self, 'legend'):
                    self.legend.addItem(series, name=c)

        self.plot.showGrid(x = True, y = True, alpha = 1)
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.disableAutoRange()
        self.plot.hideButtons()

        for axis in ['bottom', 'left']:
            ax = self.plot.getAxis(axis)
            ax.showLabel(False)
            ax.showLabel(False)
            ax.setStyle(tickLength=0, showValues=False)

        self.ignore_hover = [x, y]

        self.plot.setRange(xRange=(-2,2), yRange=(-2,2), padding=0, disableAutoRange=True)

    def on_click(self, plot, points):
        if not self.selectable:
            return
        if isinstance(points, pg.graphicsItems.ScatterPlotItem.SpotItem):
            points = [points]

        for p in self.last_clicked:
            if p is not None:
                p.resetPen()
        for p in points:
            if p is not None:
                p.setPen('b', width=1)
        self.last_clicked = points

    def on_move(self, pos):
        current_hovered = []
        new_hovered     = []
        hover_exits     = self.last_hovered

        for key in self.series:
            act_pos = self.series[key].mapFromScene(pos)

            series_hovered  = self.series[key].pointsAt(act_pos)
            current_hovered = current_hovered + series_hovered

            for point in series_hovered:
                # If point not hovered in last frame, call on_hover_enter
                if point not in self.last_hovered:
                    new_hovered.append(point)
                # If point was already hovered in last frame, remove it
                # This is so self.last_hovered becomes a list of points that
                # were hovered last frame, but not this frame, so on_hover_exit
                # should be called on them
                else:
                    hover_exits.remove(point)
        
        # Call on_hover_enter for new hovers
        for point in new_hovered:
            self.on_hover_enter(point)

        # Call on_hover_exit for old hovers that aren't currently hovered
        for point in hover_exits:
            self.on_hover_exit(point)

        if len(current_hovered) == 0:
            self.hide_tooltip()

        self.last_hovered = current_hovered

    def display_tooltip(self, pos=None, text=None, html=None):
        if not hasattr(self, 'tooltip'):
            return

        if self.tooltip not in self.plot.items:
            self.plot.addItem(self.tooltip)
        self.tooltip.setVisible(True)

        if html is not None:
            self.tooltip.setHtml(html)
        elif text is not None:
            self.tooltip.setText(text)
        
        if pos is not None:
            self.tooltip.setPos(pos.x(), pos.y())

    def hide_tooltip(self):
        if not hasattr(self, 'tooltip') or self.tooltip not in self.plot.items:
            return

        self.tooltip.setVisible(False)

    def on_hover_enter(self, point):

        data = point._data[6]
        html = ""

        #metrics = [key for key in data if key not in [self.category_col, self.x_col, self.y_col, "color"]]
        metrics = [key for key in data if key not in [self.category_col, "color"] + self.ignore_hover]

        if self.category_col is not None and self.category_col != "":
            html += f"<h2 style='color: rgb{data['color']}'>{data[self.category_col]}</h2>"

        #html += f"<p><b>{self.x_col}:</b> {data[self.x_col]}"
        #html += f"<p><b>{self.y_col}:</b> {data[self.y_col]}"

        if len(metrics) > 0:
            #html += "<br>"
            for metric in metrics:
                html += f"<p><b>{elide_text(metric)}:</b> {data[metric]}"

        self.display_tooltip(point.viewPos(), html=html)

    def on_hover_exit(self, point):
        pass

    def on_items_selected(self, items):
        self.current_selection = items
        self.on_click(self.plot, items)

class CustomPlot():
    def __init__(self, parent=None, controller=None):
        self.parent = parent
        self.controller = controller
        self.view = pg.GraphicsLayoutWidget()
        self.plot = self.view.addPlot()
        #self.view.sceneObj.sigMouseMoved.connect(self.on_move)

    def histogram(self, vals):
        self.plot.clear()
        try:
            y,x = np.histogram(vals, bins=10)
            self.plot_item = self.plot.plot(x,y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
            self.plot.setLabel('left', 'Count')
            self.plot.setLabel('bottom', 'Bucket')
            self.plot_item.scatter.sigClicked.connect(self.on_click)
        except ValueError as e:
            print(str(e) + " ---- Occurred in CustomPlot's histogram() method.")

    def line_graph(self, x, y, string_x=True):
        self.plot.clear()
        if not string_x:
            self.plot_item = self.plot.plot(x, y, pen=pg.mkPen((255,0,0)))
            return

        xdict = dict(enumerate(x))
        self.plot.plot(list(xdict.keys()), y, pen=pg.mkPen(color=(80,50,255), width=3))

        string_axis = AxisItemRotate(orientation='bottom', degrees=90)
        string_axis.setTicks([xdict.items()])

        self.plot.setAxisItems({'bottom': string_axis})

    def bar_graph(self, x, y, string_x=True):
        self.plot.clear()
        if not string_x:
            self.bgi = BarGraphClick(x=x, height=y, width=1)
            self.plot.addItem(self.bgi)
            return

        xdict = dict(enumerate(x))
        self.bgi = BarGraphClick(x=list(xdict.keys()), height=y, width=1, pen=pg.mkPen(color=(0,0,0), width=2), brush=pg.mkBrush((80,50,255)))
        self.plot.addItem(self.bgi)

        string_axis = AxisItemRotate(orientation='bottom', degrees=90)
        string_axis.setTicks([xdict.items()])
        self.plot.setAxisItems({'bottom': string_axis})

    def clear_plot(self):
        self.plot.clear()

    def on_click(self, points):
        print(points)

    def on_move(self, pos):
        print(pos)

class AxisItemRotate(pg.AxisItem):
    
    def __init__(self, orientation, pen=None, textPen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True, text='', units='', unitPrefix='', degrees=0, **args):
        self._degrees = degrees
        self._height_updated = False
        super(AxisItemRotate, self).__init__(orientation, pen=None, textPen=None, linkView=None, parent=None, maxTickLength=-5, showValues=True, text='', units='', unitPrefix='', **args)

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)

        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)

        # Draw all text
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        p.setPen(self.textPen())

        max_width = 0
        for rect, flags, text in textSpecs:
            p.save()  # save the painter state

            p.translate(rect.center())   # move coordinate system to center of text rect
            p.rotate(self._degrees)  # rotate text
            p.translate(-rect.center())  # revert coordinate system

            x_offset = math.ceil(math.fabs(math.sin(math.radians(self._degrees)) * rect.width()))
            p.translate(x_offset/2, 0)  # Move the coordinate system (relatively) downwards

            p.drawText(rect, flags, text)
            p.restore()  # restore the painter state
            offset = math.fabs(x_offset)
            max_width = offset if max_width < offset else max_width

        #  Adjust the height
        if not self._height_updated:
            self.setHeight(self.height() + max_width)
            self._height_updated = True

        #profiler('draw text')

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, controller=None, width=5, height=4, dpi=100):
        self.parent = parent
        self.controller = controller
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        #self.fig.set_facecolor("black")

    def clear_plot(self):
        self.axes.cla()

    def boxplot(self, vals):
        if vals is None:
            self.clear_plot()
            return

        data = []
        if isinstance(vals, list):
            for val in vals:
                data.append(generate_box_dict(val))
        else:
            data.append(generate_box_dict(vals))

        self.axes.bxp(data, showfliers=False)
        self.draw()

class BarGraphClick(pg.BarGraphItem):
    def __init__(self, *args, **kwargs):
        super(BarGraphClick, self).__init__(*args, **kwargs)
        self.acceptHoverEvents()

    def hoverEnterEvent(self, event):
        print(event)

    def mouseClickEvent(self, event):
        print(event)
