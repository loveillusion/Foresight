import pandas as pd
import argparse
import csv
import joblib


def main(province, NoH, Pop):
    provinces = ['Header', 'AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'ON', 'PE', 'QC', 'SK']
    columns = ['prov', 'min_NoH', 'max_NoH', 'min_Pop', 'max_Pop', 'min_NoB', 'max_NoB', 'min_Disc', 'max_Disc']

    # Get Required Row:
    maxmin = open('Foresight/datasets/maxmin.csv')
    maxmin = csv.reader(maxmin)
    rows = []
    for row in maxmin:
        rows.append(row)
    maxmin = rows[provinces.index(province)]

    # Loading Models:
    m1 = joblib.load('Foresight/models/' + province + '/m1.sav')
    m2 = joblib.load('Foresight/models/' + province + '/m2.sav')

    # Normalization:
    NoH = (NoH - int(maxmin[columns.index('min_NoH')])) / (
                int(maxmin[columns.index('max_NoH')]) - int(maxmin[columns.index('min_NoH')]))
    Pop = (Pop - int(maxmin[columns.index('min_Pop')])) / (
                int(maxmin[columns.index('max_Pop')]) - int(maxmin[columns.index('min_Pop')]))

    # Run Prediction:
    dataframe1 = pd.DataFrame([[NoH, Pop]], columns=['Number of Homes', 'Population'])
    NoB = (m1.predict(dataframe1))[0][0]
    dataframe2 = pd.DataFrame([[NoH, Pop, NoB]], columns=['Number of Homes', 'Population', 'Number of Beds'])
    Disc = (m2.predict(dataframe2))[0][0]

    # Find Actual Prediction:
    NoB = NoB * (int(maxmin[columns.index('max_NoB')]) - int(maxmin[columns.index('min_NoB')])) + int(
        maxmin[columns.index('min_NoB')])
    Disc = Disc * (int(maxmin[columns.index('max_Disc')]) - int(maxmin[columns.index('min_Disc')])) + int(
        maxmin[columns.index('min_Disc')])
    prediction = int(NoB + Disc)
    print(prediction)


def parse_input():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--province", required=True, help="Alpha Code of Province", type=str, choices=["AB", "BC", "MB", "NB", "ON"]
    )
    arg_parser.add_argument(
        "--NoH", required=False, help="Number of Homes", type=int
    )
    arg_parser.add_argument(
        "--Pop", required=False, help="Population", type=int
    )
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_input()
    main(args.province, args.NoH, args.Pop)

