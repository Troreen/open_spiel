import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_results():
    """Plot results from the NFSP dark hex experiment."""
    # load the data
    with open('tmp/nfsp_test_3x3_resnet_checkpoints/rand_game_results.pkl', 'rb') as f:
        data = pickle.load(f)
        
    # data = {
    #           "rand_game_results": rand_game_results,
    #           "num_train_episodes": FLAGS.num_train_episodes,
    #           "eval_every": FLAGS.eval_every,
    #           "num_eval_games": FLAGS.num_eval_games,
    #       }
        
    results = data["rand_game_results"]
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
    
    
if __name__ == '__main__':
    plot_results()
