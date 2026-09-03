import pandas as pd
from datetime import date
import calendar

def third_wednesday(year, month):
    c = calendar.Calendar()
    wednesdays = [d for d in c.itermonthdates(year, month)
                  if d.month == month and d.weekday() == 2]
    return wednesdays[2]

def is_settlement_day(ts):
    d = ts.date()
    tw = third_wednesday(d.year, d.month)
    return d == tw

if __name__ == '__main__':
    print(third_wednesday(2026, 6))
    print(third_wednesday(2020, 1))
