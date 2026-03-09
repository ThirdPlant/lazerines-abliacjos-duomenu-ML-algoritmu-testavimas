import pandas as pd

df = pd.read_excel("Profiler data all1.xlsx", sheet_name="CuN1")
df_P = pd.DataFrame()
df_N = pd.DataFrame()

def vienas(df, k):
    df_P = pd.DataFrame()
    for i in range(24):
        pirmas = (3+14*i)
        antras = (14+14*i)
        Nscan = df.iloc[pirmas:antras, 15]
        F0 = df.iloc[pirmas:antras, 14]

        Rate1 = df.iloc[pirmas:antras, 18]
        Eff = df.iloc[pirmas:antras, 20]
        Ra = df.iloc[pirmas:antras, 17]
        Gylis = df.iloc[pirmas:antras, 16]

        df2 = pd.DataFrame(
            {
                "Nscan":Nscan,
                "F0":F0,
                "Rate1":Rate1,
                "Ra":Ra,
                "Gylis":Gylis
            }
        )
        df2.insert(0, "P", i+2)
        df2.insert(0, "N", k)
        df_P = pd.concat([df_P, df2], axis=0, ignore_index=True)

    df_P.loc[df_P["P"] == 11, "P"] = 15
    df_P.loc[df_P["P"] == 12, "P"] = 20
    df_P.loc[df_P["P"] == 13, "P"] = 25
    return df_P

for k in range(9):
    k=k+1
    df = pd.read_excel("Profiler data all1.xlsx", sheet_name=f"CuN{k}")
    df_P = vienas(df, k)
    df_N = pd.concat([df_N, df_P], axis=0, ignore_index=True)
    




def vienas2(df, df_N):
    for i in range(9):
        pirmas = (3+14*i)
        antras = (14+14*i)
        Nscan = df.iloc[pirmas:antras, 15]
        F0 = df.iloc[pirmas:antras, 14]

        Rate1 = df.iloc[pirmas:antras, 18]
        Eff = df.iloc[pirmas:antras, 20]
        Ra = df.iloc[pirmas:antras, 17]
        Gylis = df.iloc[pirmas:antras, 16]

        df2 = pd.DataFrame(
            {
                "Nscan":Nscan,
                "F0":F0,
                "Rate1":Rate1,
                "Ra":Ra,
                "Gylis":Gylis
            }
        )
        df2.insert(0, "P", 1)
        df2.insert(0, "N", i+1)

        df_N = pd.concat([df_N[df_N["N"] <= i], df2, df_N[df_N["N"] >= i+1]], axis=0, ignore_index=True)
    return df_N

df_P1 = pd.read_excel("Profiler data all1.xlsx", sheet_name="CuP1")

df_N = vienas2(df_P1, df_N)

df_N = df_N[df_N["Gylis"] > 0]


df_N.to_excel("surikiuoti_duomenys.xlsx", index=False)

