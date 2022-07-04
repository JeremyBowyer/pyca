from PyQt5.QtCore import pyqtSlot, QVariant, QObject


class CallHandler(QObject):
    # Call method from JS like this: handler.js_to_python_to_js('hi', function(retVal) { console.error(JSON.stringify(retVal)); })
    # https://doc.qt.io/qt-5/qtwebchannel-javascript.html
    @pyqtSlot(QVariant, result=QVariant) # Javascript object looking for signature based on pyqtSlot
    def js_to_python_to_js(self, args):
        print('call received')
        print(args)
        return QVariant({"abc": "def", "ab": 22})

    @pyqtSlot(QVariant) # Javascript object looking for signature based on pyqtSlot
    def js_to_python(self, args):
        print("js_to_python")
        print(args)


class PlotlyHandler(CallHandler):
    def __init__(self):
        super(PlotlyHandler, self).__init__()
        self.__on_hover     = None
        self.__on_unhover   = None
        self.__on_select    = None
        self.__on_deselect  = None
        self.__on_click     = None

    def set_on_hover(self, cb):
        self.__on_hover = cb

    def set_on_unhover(self, cb):
        self.__on_unhover = cb

    def set_on_select(self, cb):
        self.__on_select = cb
        
    def set_on_deselect(self, cb):
        self.__on_deselect = cb

    def set_on_click(self, cb):
        self.__on_click = cb

    @pyqtSlot(QVariant)
    def on_hover(self, event_data):
        if callable(self.__on_hover):
            self.__on_hover(event_data)

    @pyqtSlot(QVariant)
    def on_unhover(self, event_data):
        if callable(self.__on_unhover):
            self.__on_unhover(event_data)

    @pyqtSlot(QVariant)
    def on_select(self, event_data):
        if event_data is None:
            self.on_deselect()
            return
        if callable(self.__on_select):
            self.__on_select(event_data)

    # Called if selection is None
    def on_deselect(self):
        if callable(self.__on_deselect):
            self.__on_deselect()

    @pyqtSlot(QVariant)
    def on_click(self, event_data):
        if callable(self.__on_click):
            self.__on_click(event_data)