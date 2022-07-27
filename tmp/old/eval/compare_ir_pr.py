import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.color_palette("tab10")
sns.set_style("darkgrid")


def compare_results_random(early_terminal, no_early_terminal):
    with open(f"{early_terminal}/rand_res.pkl", "rb") as file:
        rand_res_0 = pickle.load(file)
    
    with open(f"{no_early_terminal}/rand_res.pkl", "rb") as file:
        rand_res_1 = pickle.load(file)

    p0_results_0 = [res[0] for res in rand_res_0]
    p1_results_0 = [res[1] for res in rand_res_0]

    p0_results_1 = [res[0] for res in rand_res_1]
    p1_results_1 = [res[1] for res in rand_res_1]

    # plot mccfr results against random for each player as a subplot
    fig, ax = plt.subplots(1, 2)
    plt.title('MCCFR 4x3 Dark Hex Comparison (Early Terminal vs No-Early Terminal)')
    ax[0].plot(p0_results_0, label='Early Terminal')
    ax[0].plot(p0_results_1, label='No-Early Terminal')
    ax[0].set_xlabel(f'Iteration (x{int(5e5)})')
    ax[0].set_ylabel('Win rate')
    ax[0].title.set_text('P0 win rate against random')

    ax[1].plot(p1_results_0, label='Early Terminal')
    ax[1].plot(p1_results_1, label='No-Early Terminal')
    ax[1].set_xlabel(f'Iteration (x{int(5e5)})')
    ax[1].set_ylabel('Win rate')
    ax[1].title.set_text('P1 win rate against random')

    # legend
    ax[0].legend()
    ax[1].legend()

    # save
    plt.savefig("tmp/mccfr_dark_hex_comparison.png")


def compare_results_random_nfsp(early_terminal, no_early_terminal):
    with open(f"{early_terminal}/game_res.pkl", "rb") as file:
        rand_res_0 = pickle.load(file)
    eval_per = rand_res_0["eval_every"]
    rand_res_0 = rand_res_0["game_res"]
    
    with open(f"{no_early_terminal}/game_res.pkl", "rb") as file:
        rand_res_1 = pickle.load(file)
    rand_res_1 = rand_res_1["game_res"]
    
    p0_results_0 = [res[0] for res in rand_res_0]
    p1_results_0 = [res[1] for res in rand_res_0]

    p0_results_1 = [res[0] for res in rand_res_1]
    p1_results_1 = [res[1] for res in rand_res_1]

    # plot mccfr results against random for each player as a subplot
    fig, ax = plt.subplots(1, 2)
    plt.title('NFSP 4x3 Dark Hex Comparison (Early Terminal vs No-Early Terminal)')
    ax[0].plot(p0_results_0, label='Early Terminal')
    ax[0].plot(p0_results_1, label='No-Early Terminal')
    ax[0].set_xlabel(f'Iteration (x{eval_per})')
    ax[0].set_ylabel('Win rate')
    ax[0].title.set_text('P0 win rate against random')

    ax[1].plot(p1_results_0, label='Early Terminal')
    ax[1].plot(p1_results_1, label='No-Early Terminal')
    ax[1].set_xlabel(f'Iteration (x{eval_per})')
    ax[1].set_ylabel('Win rate')
    ax[1].title.set_text('P1 win rate against random')

    # legend
    ax[0].legend()
    ax[1].legend()

    # save
    plt.savefig("tmp/nfsp_dark_hex_comparison.png")


def compare_results_nashconv_nfsp(early_terminal, no_early_terminal):
    with open(f"{early_terminal}/game_res.pkl", "rb") as file:
        game_res_0 = pickle.load(file)
    eval_per = game_res_0["eval_every"]
    nash_conv_0 = game_res_0["game_res"]
    
    with open(f"{no_early_terminal}/game_res.pkl", "rb") as file:
        game_res_1 = pickle.load(file)
    nash_conv_1 = game_res_1["game_res"]

    # plot nfsp nashconv results (single figure)
    figure = plt.figure()
    plt.title('NFSP 2x2 Dark Hex')
    plt.plot(nash_conv_0, label='Early Terminal')
    plt.plot(nash_conv_1, label='No-Early Terminal')
    plt.xlabel(f'Iteration (x{eval_per})')
    plt.ylabel('NashConv')
    plt.legend()
    
    # save
    plt.savefig("tmp/nfsp_dark_hex_comparison_nashconv.png")


def compare_results_nashconv_nfsp_pr_ir(pr, ir):
    with open(f"{ir}/game_res.pkl", "rb") as file:
        ir_res = pickle.load(file)
    eval_per = ir_res["eval_every"]
    nash_conv_ir = ir_res["game_res"]
    
    with open(f"{pr}/game_res.pkl", "rb") as file:
        pr_res = pickle.load(file)
    nash_conv_pr = pr_res["game_res"]

    # plot nfsp nashconv results (single figure)
    figure = plt.figure()
    # bold title
    plt.title('NFSP 2x2 PR vs IR', fontweight='bold')
    plt.plot(nash_conv_ir, label='Imperfect Recall')
    plt.plot(nash_conv_pr, label='Perfect Recall')
    plt.xlabel(f'Iteration (x{eval_per})')
    plt.ylabel('NashConv')
    plt.legend()
    
    # save
    plt.tight_layout()
    plt.savefig("tmp/nfsp_ir_pr_comparison_nashconv.pdf")


def compare_vanilla_cfr_ir_pr(pr_path, ir_path):
    with open(pr_path, "rb") as file:
        pr_res = pickle.load(file)
    with open(ir_path, "rb") as file:
        ir_res = pickle.load(file)

    figure = plt.figure()
    plt.title('2x2 Dark Hex Vanilla CFR')
    plt.plot(ir_res, label='Imperfect Recall')
    plt.plot(pr_res, label='Perfect Recall')
    plt.xlabel(f'Iteration (x10)')
    plt.ylabel('NashConv')
    plt.legend()

    # save
    plt.savefig("tmp/vanilla_cfr_dark_hex_comparison.png")


if __name__ == "__main__":
    # compare_results_nashconv_nfsp("tmp/nfsp_test_2x2_early", "tmp/nfsp_test_2x2_no_early")
    compare_results_nashconv_nfsp_pr_ir("tmp/dark_hex_mccfr_3x2_pr", "tmp/dark_hex_mccfr_3x2_ir")