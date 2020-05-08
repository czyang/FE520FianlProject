from tabulate import tabulate
from datetime import date
from datetime import datetime

class Summary:
    def __init__(self):
        self.title = ""
        self.dataTable = []

    # Set title for the form
    def setTitle(self, title):
        self.title = title

    # Append current date to data table
    def appendDate(self):
        self.append("Date:", date.today())

    # Append current time to data table
    def appendTime(self):
        self.append("Time:", datetime.now().strftime("%H:%M:%S"))

    # Append a item (key and value) to data table
    def append(self, field, value):
        self.dataTable.append([field, value])

    # Get the summary string from data table
    def get_summary(self):
        res = "\n"
        res += self.title + "\n"
        res += tabulate(self.dataTable, tablefmt="psql", colalign=("left", "right")) + "\n"
        return res

# Usage:
# s = Summary()
# s.setTitle("Test title")
# s.append("Dep. Variable:", 123)
# s.append("bb", 234)
# print(s.get_summary())
