import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

sns.set_style("darkgrid")

def heat_map(df, num_players, title_given="Arena", save_path="tmp/Arena/res/arena.pdf"):
    mask = np.zeros_like(df)
    mask[:, num_players] = 1
    mask[num_players, :] = 1
    plt.subplots(figsize=(10,5))
    sns.heatmap(df, mask=mask, cmap="Reds", annot=True, annot_kws={"size": 11, "color":"g"}, square=False,
                xticklabels=df.columns, fmt='.3g', yticklabels=df.index, linewidths=0.5)
    sns.heatmap(df, alpha=0, cbar=False, annot=True, square=False,
                annot_kws={"size": 11, "color":"g"},
                xticklabels=df.columns, fmt='.3g', yticklabels=df.index, linewidths=0.5)
    plt.tick_params(axis='both', which='major', labelsize=13, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.xticks(rotation=90)
    plt.title(title_given, fontsize=18)
    plt.xlabel("Second Player", fontsize=15)
    plt.ylabel("First Player", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path)


def heat_map_driver(records, title, ignore_columns=None, show_ratio=False,
                    num_games=None, save_to_path="tmp/Arena/res/"):
    player_names = list(records.keys())
    if ignore_columns:
        player_names = [name for name in player_names if name not in ignore_columns]
    num_players = len(player_names)
    df = pd.DataFrame(index=player_names + ["(-)Total"], columns=player_names + ["Total"], dtype=int)
    # plot heat map
    for i in range(num_players):
        for j in range(num_players):
            if i == j:
                df.iloc[i, j] = 0
                continue
            p0_name = player_names[i]
            p1_name = player_names[j]
            p0_wins = records[p0_name][p1_name]
            df.iloc[i, j] = p0_wins
    if show_ratio:
        if not num_games:
            raise ValueError("num_games must be provided if show_ratio is True")
        # convert to average value
        df = df.divide(num_games)
    # calculate the totals
    for i in range(num_players):
        if show_ratio:
            df.iloc[i, num_players] = df.iloc[i, :].sum() / (num_players - 1)
        else:
            df.iloc[i, num_players] = df.iloc[i, :].sum()
    for j in range(num_players):
        if show_ratio:
            df.iloc[num_players, j] = -df.iloc[:, j].sum() / (num_players - 1)
        else:
            df.iloc[num_players, j] = -df.iloc[:, j].sum()
    df.iloc[num_players, num_players] = df['Total'].sum()
    
    # sort the players
    df_sorted = df.sort_values(by=df.columns[num_players], ascending=False)
    player_names.sort(key=lambda x: df_sorted[x]["(-)Total"], reverse=True)
    df_sorted = df_sorted.reindex(player_names + ["Total"], axis=1)

    if show_ratio:
        df_sorted.iloc[num_players, num_players] = (df['Total'].sum() - df['Total']["(-)Total"]) / (num_players)

    # Save the results to a csv file
    if not os.path.exists(save_to_path):
        os.makedirs(save_to_path)
    df_sorted.to_csv(save_to_path+"arena.csv")
    print("\033[1m\033[32m" + "Results saved to " + save_to_path + "\033[0m")
    heat_map(df_sorted, num_players, title_given=title, save_path=save_to_path + "arena.pdf")


if __name__ == "__main__":
    with open("tmp/Arena/res/records.pkl", "rb") as f:
        records = pickle.load(f)
    # heat_map_driver(
    #     records=records,
    #     title="Arena (Average Reward)",
    #     ignore_columns=["SIMCAP", "SIMCAP+", "SIMCAP-L", "SIMCAP-L+"],
    #     show_ratio=True,
    #     num_games=5000,
    #     save_to_path="tmp/Arena/res/no_simcap/"
    # )
    # heat_map_driver(
    #     records=records,
    #     title="Arena (Average Reward)",
    #     ignore_columns=["SIMCAP+", "SIMCAP-L+"],
    #     show_ratio=True,
    #     num_games=5000,
    #     save_to_path="tmp/Arena/res/no_simcap+/"
    # )
    heat_map_driver(
        records=records,
        title="Arena (Average Reward)",
        # ignore_columns=["SIMCAP+", "SIMCAP-L+"],
        show_ratio=True,
        num_games=5000,
        save_to_path="tmp/Arena/res/simcap_test/"
    )