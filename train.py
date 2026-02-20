import numpy as np
import wandb
import optuna

from qlearning import Qlearningagent
from oneEV import EVChargingEnv
from nEV import MultiEVChargingEnv


def objective(trial):
    # 2. Initialize a W&B run for this specific trial
    lr = trial.suggest_float("learning_rate", 0.0003, 0.5, log=True)
    gamma = trial.suggest_float("discount_factor", 0.90, 0.99)
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
            "num_episodes": 5000,
        },
        reinit=True
    )
    config = wandb.config

    env = EVChargingEnv(t_max=100)

    agent = Qlearningagent(
        state_space_size=3, 
        action_space_size=env.action_space.n,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.epsilon,
        epsilon_decay=1
    )

    total_rewards = []

    for episode in range(config.num_episodes):
        obs, _ = env.reset()
        state = int(obs)
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            next_state = int(next_obs)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

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

    final_performance = np.mean(total_rewards[-100:])
    run.finish() # 4. Close the run
    return final_performance

if __name__ == "__main__":
    # Ensure you are logged in: wandb login
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

