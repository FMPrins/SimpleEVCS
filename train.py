import numpy as np
import wandb
import optuna
import matplotlib.pyplot as plt

from qlearning import Qlearningagent
from oneEV import EVChargingEnv
from nEV import MultiEVChargingEnv


def encode_state(obs, base=3):
    """
    Encodes a multi-EV state into a single integer.
    Works for any number of EVs.
    
    obs: array-like of length n
    base: number of discrete states per EV (default=3)
    """
    obs = np.atleast_1d(obs)

    n = len(obs)

    index = 0
    base = 3  # since each EV has 3 possible states (0,1,2)

    for i in range(n):
        index += obs[i] * (base ** i)

    return int(index)

def encode_action(action_vec, I_max):
    """
    action_vec: array of length n_evs
    """
    base = I_max + 1
    index = 0
    n = len(action_vec)

    for i in range(n):
        index += action_vec[i] * (base ** (n - i - 1))

    return index

def decode_state(index, n_evs, base=3):
    """
    Decodes integer index back into EV state vector.
    """
    obs = np.zeros(n_evs, dtype=int)
    
    for i in range(n_evs - 1, -1, -1):
        obs[i] = index % base
        index //= base
        
    return obs

def decode_action(index, n_evs, I_max):
    """
    Converts integer back into action vector.
    """
    base = I_max + 1
    action = np.zeros(n_evs, dtype=int)

    for i in range(n_evs - 1, -1, -1):
        action[i] = index % base
        index //= base

    return action


def objective(trial):
    # 2. Initialize a W&B run for this specific trial
    lr =  0.03 #trial.suggest_float("learning_rate", 0.01, 0.05, log=True)
    gamma = 0.99 #trial.suggest_float("discount_factor", 0.90, 0.99)
    eps = 0.01 
    
    # 2. Create a descriptive name
    # Format: "run_LR-0.01_G-0.95_ED-0.995"
    run_name = f"trial_{trial.number}_LR-{lr:.3f}_G-{gamma:.2f}_E-{eps:.4f}"

    # 3. Initialize W&B with the custom name
    run = wandb.init(
        project="ev-charging-optimization",
        group="optuna-study",
        name=run_name,  
        config={
            "learning_rate": lr,
            "discount_factor": gamma,
            "epsilon": eps,
            "trial_num": trial.number,
            "num_episodes": 10000,
        },
        reinit=True
    )
    config = wandb.config

    n_evs = 1
    i_max = 1

    # Multiple EVs, also allows for one EV. 
    env = MultiEVChargingEnv(t_max=100, i_max = i_max, p_max = 1, n_evs= n_evs)

    #Turn on for the old environment with only one EV. 
    #env = EVChargingEnv(t_max = 100, i_max = i_max)

    agent = Qlearningagent(
        state_space_size = 3 ** n_evs,              #3 ** env.n_evs, 
        action_space_size= (i_max + 1) ** n_evs,    #(env.I_max + 1) ** env.n_envs,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.epsilon,
        epsilon_decay=1
    )

    total_rewards = []

    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        state = encode_state(obs)
        episode_reward = 0
        done = False

        while not done:
            action_index = agent.select_action(state)

            next_obs, reward, terminated, truncated, _ = env.step(action_index)

            next_state = encode_state(next_obs)
            done = terminated or truncated

            agent.update(state, action_index, reward, next_state, done)

            state = next_state
            episode_reward += reward

        #agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        total_rewards.append(episode_reward)

        # 3. Log metrics to W&B
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])


            best_action_state_1 = np.argmax(agent.q_table[1])
            
            wandb.log({
                "episode": episode + 1,
                "avg_reward_100": avg_reward,
                "epsilon": agent.epsilon,
                "best_action_state_1": best_action_state_1
            })

            # Report back to Optuna for pruning
            # trial.report(avg_reward, episode)
            # if trial.should_prune():
            #     run.finish(exit_code=1) # Mark as failed/pruned in W&B
            #     raise optuna.exceptions.TrialPruned()

            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(agent.q_table, aspect='auto')
            plt.colorbar()
            plt.title(f"Q-table Heatmap (Episode {episode+1})")
            plt.xlabel("Actions")
            plt.ylabel("States")

            wandb.log({
                "Q_table_heatmap": wandb.Image(plt)
            })

            plt.close()

    best_actions_idx = np.argmax(agent.q_table, axis=1)

    policy_table = wandb.Table(columns=["state_vector", "best_action"])

    for state_idx, action_idx in enumerate(best_actions_idx):

        state_vec = decode_state(state_idx, env.n_evs)
        action_vec = decode_action(action_idx, env.n_evs, env.I_max)

        policy_table.add_data(str(state_vec), str(action_vec))

    wandb.log({"policy_table": policy_table})

    final_performance = np.mean(total_rewards[-100:])
    run.finish() # 4. Close the run
    return final_performance

if __name__ == "__main__":
    # Ensure you are logged in: wandb login
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

