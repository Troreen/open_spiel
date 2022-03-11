import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_results_random(data):
    results = data["rand_res"]
    eval_every = data["eval_every"]
    
    p0_results = [res[0] for res in results]
    p1_results = [res[1] for res in results]
    
    # plot the results in two subplots
    fig, ax = plt.subplots(1, 2)
    plt.title('NFSP 4x3 Dark Hex')
    ax[0].plot(p0_results)
    ax[0].set_xlabel(f'Iteration (x{eval_every})')
    ax[0].set_ylabel('Win rate')
    ax[0].title.set_text('P0 win rate against random')
    
    ax[1].plot(p1_results)
    ax[1].set_xlabel(f'Iteration (x{eval_every})')
    ax[1].set_ylabel('Win rate')
    ax[1].title.set_text('P1 win rate against random')
    
    plt.savefig('tmp/nfsp_dark_hex.png')
    
def plot_results_nashconv(data):
    results = data["game_res"]
    eval_every = data["eval_every"]
    
    plt.title('NFSP 2x2 Dark Hex')
    plt.plot(results)
    plt.xlabel(f'Iteration (x{eval_every})')
    plt.ylabel('NashConv')
    plt.savefig('tmp/nfsp_dark_hex_nc.png')
    
    
if __name__ == '__main__':
    with open('tmp/nfsp_test_2x2_mlp_ir_checkpoints/game_res.pkl', 'rb') as f:
        data = pickle.load(f)
    plot_results_nashconv(data)
    plot_results_random(data)
