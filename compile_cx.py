import sys, os
from cx_Freeze import setup, Executable

os.environ['TCL_LIBRARY'] = r'F:\Programs\Python\Python36\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'F:\Programs\Python\Python36\tcl\tk8.6'

__version__ = "1.1.0"

include_files = ["map.jpg", "tcl86t.dll", "tk86t.dll"]
excludes = []
packages = ["tkinter", "PIL"]
base = 'Win32GUI' if sys.platform=='win32' else None
executables = [Executable("app.py",base=base)]

setup(
    name = "FnD",
    description="A Fallout and Dragons Character App",
    version=__version__,
    options = {"build_exe": {
    'packages': packages,
    'include_files': include_files,
    'excludes': excludes,
    'include_msvcr': True,
}},
executables = executables
)