import pandas as pd
import os
from tqdm import tqdm


def csv2wkt(filepath):
    savepath = os.path.dirname(filepath)
    filename = os.path.basename(filepath).split(".")[0] + "_out.csv"
    # csvShape = r"E:\PythonProject\shp2wkt\work\listings.csv"
    df = pd.read_csv(filepath, sep=',', encoding='gbk', engine='python')
    df.drop('url', axis=1, inplace=True)
    # df.drop(df.columns[2], axis=1, inplace=True)
    # df.drop(df.columns[3], axis=1, inplace=True)
    # df.to_csv('./test.csv', header=None, sep='\t')
    list_table = list()
    lat = df['Lat']
    lng = df["Lng"]

    i = 0
    for la in tqdm(lat):
        list_table.insert(i, "POINT (" + str(lng[i]) + " " + str(la) + ")")
        i += 1

    df.insert(16, "polygon", list_table)
    # print(df)
    df.to_csv(os.path.join(savepath, filename), header=None, sep='\t')


if __name__ == '__main__':
    path = r"G:\temp\data\ori\listings.csv"
    csv2wkt(path)