import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_results_random(data_folder):
    with open(f"{data_folder}/rand_res.pkl", "rb") as file:
        rand_res = pickle.load(file)

    p0_results = [res[0] for res in rand_res]
    p1_results = [res[1] for res in rand_res]

    # plot mccfr results against random for each player as a subplot
    fig, ax = plt.subplots(1, 2)
    plt.title('MCCFR 4x3 Dark Hex')
    ax[0].plot(p0_results)
    ax[0].set_xlabel(f'Iteration (x{int(25e3)})')
    ax[0].set_ylabel('Win rate')
    ax[0].title.set_text('P0 win rate against random')

    ax[1].plot(p1_results)
    ax[1].set_xlabel(f'Iteration (x{int(25e3)})')
    ax[1].set_ylabel('Win rate')
    ax[1].title.set_text('P1 win rate against random')

    plt.savefig(f"{data_folder}/mccfr_dark_hex.png")


if __name__ == "__main__":
    plot_results_random("tmp/dark_hex_mccfr_4x3")