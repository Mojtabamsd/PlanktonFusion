import pandas as pd


def report_to_df(report):
    report = [x.split(" ") for x in report.split("\n")]
    header = ["Class Name"] + [x for x in report[0] if x != ""]
    values = []
    for row in report[1:-1]:
        row = [value for value in row if value != ""]
        if row != []:
            while row.__len__() > header.__len__():
                tmp = list([row[0] + ' ' + row[1]])
                new_row = tmp + row[2:]
                row = new_row
            values.append(row)
    df = pd.DataFrame(data=values, columns=header)
    return df