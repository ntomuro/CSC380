"""
main.py

CSC 380 Frozen Lake, Graphic display demo

This code can be called in these days:
- 'python main.py' -- episodes will run using a random fixed policy
- 'python main.py policy.txt' -- episodes will run using a give policy stored
                                   in a file, where a policy is written in one
                                   line separated by spaces, e.g.
                                   '3 1 0 3 2 ...'

The number of episodes per run (2 episodes for now) can be specified by
giving a value to the variable 'num_episodes' (inside main()).

"""
import gymnasium as gym
import numpy as np
import math
import time
import sys

def display_policy(policy, nS):
    side = int(math.sqrt(nS))  # assuming a square
    policy = policy.reshape((side, side))
    return policy

def generate_random_policy(num_actions, num_states, seed=None):
    """
    A policy is a 1D array of length # of states, where each element is a
    number between 0 (inclusive) and # of actions (exclusive) randomly chosen.
    If a specific seed is passed, the same numbers are genereated, while
    if the seed is None, the numbers are unpredictable every time.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=num_actions, size=num_states)


def run_oneexperiment(env, policy, num_episodes, display=False):
    """
    Run one experiment, when agent follows a policy, for a given number of episodes.
    """
    # Count the number of goals made and getting stuck in a hole
    goals = 0
    holes = 0
    # Total rewards and steps
    total_rewards = 0
    total_goal_steps = 0

    # This line is critically needed to access the state and the probabilities in the environment
    #  due to the recent code update by Gymnasium.
    env_unwrapped = env.unwrapped

    for _ in range(num_episodes):
        # For each time,
        env.reset()
        done = False
        rewards = 0
        steps = 0
        reward = -1

        #if display:
        #    episode = [(env_unwrapped.env.s)] # initial state (in a tuple)

        while not done:
            # choose the action based on the policy
            state = env_unwrapped.s
            action = policy[state]  # (!) look up the policy and choose the action for the state

            # take the action
            next_state, reward, done, info, p = env.step(action)  # take the chosen action
            steps += 1

            env.render()
            time.sleep(0.2)

            # extend the episode
            #if display:
            #    episode.append(tuple([action,next_state]))
            # accumulate rewards
            rewards += reward

        # Calculate stats
        total_rewards += rewards
        if reward == 1.0: # Goal, or env.s == 63
            goals += 1
            total_goal_steps += steps
        else:
            holes += 1

        # Display
        if display:
            env.render()

    # One experiment finished,
    return goals, holes, total_rewards, total_goal_steps

#====================================
def main(fname:str = None):
    #Create a FrozenLake 8x8 environment using Gymnasium
    # (https://gymnasium.farama.org/environments/toy_text/frozen_lake/).
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="human")

    # Reset the environment and display it
    env.reset()
    env.render()  # wrap render() in print()

    nS = int(env.observation_space.n)  # number of states -- 8x8=64
    nA = int(env.action_space.n)  # number of actions -- four directions; 0:left, 1:down, 2:right, 3:up
    print(f"number of states: {nS}\nnumber of actions: {nA}")

    #---------------------------------
    if fname is None:
        policy = generate_random_policy(nA, nS, 7) # change seed to a specific number, or None (default)
    else:
        # read in the given policy -- # opening the CSV file
        with open(fname, mode ='r', newline='') as file:
            row = file.readline().split()
            policy = np.array([int(e) for e in row])

    print ("*** Policy ***\n{}".format(display_policy(policy, nS)))

    num_episodes = 2

    goals, holes, total_rewards, total_goal_steps \
        = run_oneexperiment(env, policy, num_episodes)

    percent_goal = goals / num_episodes
    percent_hole = holes / num_episodes
    mean_reward = total_rewards / num_episodes
    mean_goal_steps = 0.0 if (goals == 0) else (total_goal_steps / goals)

    print ("\n*** RESULTS ***:\nGoals: {:>5d}/{} = {:>7.3%}\nHoles: {:>5d}/{} = {:>7.3%}"
           .format(goals, num_episodes, percent_goal, holes, num_episodes, percent_hole))
    print("mean reward:          {:.5f}\nmean goal steps:     {:.2f}".format(mean_reward, mean_goal_steps))

##=====================================================================
if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        main(args[1])
    else:
        main()