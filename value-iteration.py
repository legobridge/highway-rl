import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class ValueIterationAgent:
    def __init__(self, env):

        # convert environment to finite MDP
        self.mdp = env.unwrapped.to_finite_mdp()
        self.env = env

        self.num_states = self.mdp.transition.shape[0]
        self.num_actions = self.env.action_space.n

        self.values = np.zeros(self.mdp.transition.shape[0])     # initialize v(s) arbitrarily for each state
        self.policy = np.zeros(self.mdp.transition.shape[0], dtype=int)     # initialize policy
        self.EPISODES = 10000
        self.gamma = 0.9
        self.epsilon = 1e-10    # a small number

    @staticmethod
    def is_finite_mdp(env):
        try:
            finite_mdp = __import__("finite_mdp.envs.finite_mdp_env")
            if isinstance(env.unwrapped, finite_mdp.envs.finite_mdp_env.FiniteMDPEnv):
                return True
        except (ModuleNotFoundError, TypeError):
            return False

    def value_iteration(self):
        delta = 0
        for i in range(self.EPISODES):
            # copy previous values
            prev_values = self.values.copy()

            for state in range(self.num_states):
                # calculate Q for each action in the state
                q_sa = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    next_state = self.mdp.next_state(state, action)     # transition[state, action]
                    r = self.mdp.reward[state, action]
                    # print(f"state {state} action {action} next_state {next_state} reward {r}")
                    q_sa[action] += (r + self.gamma * prev_values[next_state])

                # print(f"q_sa {q_sa} updated value {max(q_sa)} updated policy {np.argmax(q_sa)}")
                self.values[state] = max(q_sa)
                self.policy[state] = np.argmax(q_sa)
                # delta = max(delta, abs(prev_values[state] - self.values[state]))

            #if delta <= self.epsilon:
            if np.sum(np.fabs(prev_values - self.values)) <= 1e-18:
                print('Problem converged at iteration %d.' % (i + 1))
                break

    def policy_evaluation(self, episodes):

        for episode in range(episodes):
            done = False
            steps_survived = 0
            total_reward = 0
            self.mdp.state = np.random.choice(self.num_states)
            state = self.mdp.state

            while not done:
                action = self.policy[state]
                # print(f"state: {state} action: {action}")
                next_state, reward, done, _ = self.mdp.step(action)
                state = next_state

                steps_survived += 1
                total_reward += reward

                if done:
                    print('episode: {}/{}, steps survived: {}, total reward: {}'.format(episode + 1,
                                                                                        episodes,
                                                                                        steps_survived,
                                                                                        total_reward))
    def plot(self):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(self.values)
        ax1.set_title('Optimal Value Function')

        ax2.plot(self.policy)
        ax2.set_title('Optimal Policy')
        plt.show()

if __name__ == '__main__':

    # config = json.load(open('config.json', 'r'))
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.config['policy_frequency'] = 10

    # create value iteration agent
    agent = ValueIterationAgent(env)

    print(f"mdp mode: {agent.mdp.mode}")
    print(f"transition shape: {agent.mdp.transition.shape}")

    agent.value_iteration()
    agent.plot()
    agent.policy_evaluation(40)
