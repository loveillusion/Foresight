import pandas as pd

df = pd.read_excel('data set.xlsx')
df2 = pd.read_excel('canadacities.xlsx')
df3 = pd.read_csv('Designated places (DPLs)_Geo_starting_row_CSV.csv')
df4 = pd.read_csv('98-401-X2016047_English_CSV_data.csv')
df5 = pd.read_csv('Dissolved census subdivisions (Dissolved CSDs)_Geo_starting_row_CSV.csv')
df6 = pd.read_csv('98-401-X2016057_English_CSV_data.csv')
df7 = pd.read_csv('Canada, provinces, territories, census divisions (CDs) and census subdivisions (CSDs)_Geo_starting_row_CSV.csv')
df8 = pd.read_csv('98-401-X2016055_English_CSV_data.csv')

# print(df)

def findpop(city: str) -> int:
    for i, r in df2.iterrows():
        if r["city_ascii"].lower() == city.lower():
            return r["population"]
    for i, r in df3.iterrows():
        if r["Geo Name"].lower() == city.lower():
            linenum = r["Line Number"] - 1
            return int(df4.iloc[linenum][-3])
    for i, r in df5.iterrows():
        if r["Geo Name"].lower() == city.lower():
            linenum = r["Line Number"] - 1
            return int(df6.iloc[linenum][-3])
    for i, r in df7.iterrows():
        if r["Geo Name"].lower() == city.lower():
            linenum = r["Line Number"] - 1
            return int(df8.iloc[linenum][-3])
    return 0

# print(df4.iloc[1][-3])

# print(df['City'])

poplst = []
cnt = 0
for index, row in df.iterrows():
    num = findpop(row["City"])
    if num == 0:
        cnt += 1
    poplst.append(num)

# print(poplst)
# print(len(poplst))
# print(cnt)

df['Population'] = poplst
df.to_csv('out.csv')
