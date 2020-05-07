from tabulate import tabulate
from datetime import date
from datetime import datetime

class Summary:
    def __init__(self):
        self.title = ""
        self.dataTable = []

    def setTitle(self, title):
        self.title = title

    def appendDate(self):
        self.append("Date:", date.today())

    def appendTime(self):
        self.append("Time:", datetime.now().strftime("%H:%M:%S"))

    def append(self, field, value):
        self.dataTable.append([field, value])

    def get_summary(self):
        res = "\n"
        res += self.title + "\n"
        res += tabulate(self.dataTable, tablefmt="psql", colalign=("left", "right")) + "\n"
        return res


# s = Summary()
# s.setTitle("Test title")
# s.append("Dep. Variable:", 123)
# s.append("bb", 234)
# print(s.get_summary())
