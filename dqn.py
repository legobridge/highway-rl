import random
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam


# policy network
def create_dqn_model(input_shape, action_space):
    X_input = Input(shape=input_shape)

    X = Flatten()(X_input)

    X = Dense(128, activation='relu', kernel_initializer='he_uniform')(X)

    X = Dense(256, activation='relu', kernel_initializer='he_uniform')(X)

    X = Dense(128, activation='relu', kernel_initializer='he_uniform')(X)

    X = Dense(action_space, activation='linear', kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='DQN_model')
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002))

    model.summary()
    return model


class DQNAgent:

    def __init__(self):
        self.env = gym.make('highway-v0', render_mode='rgb_array')

        self.env.config["duration"] = 50
        self.env.config["reward_speed_range"] = [20, 60]
        self.env.config["lane_change_reward"] = 0.05

        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.EPISODES = 500
        self.memory = deque(maxlen=2**15)

        self.gamma = 0.80  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.batch_size = 256
        self.train_start = 512

        self.model = create_dqn_model(input_shape=(self.state_size[0], self.state_size[1],),
                                      action_space=self.action_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def choose_action(self, state):
        # epsilon-greedy policy
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = state.reshape([1, self.state_size[0], self.state_size[1]])
        return np.argmax(self.model.predict(state, verbose=0))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        states = np.zeros((self.batch_size, self.state_size[0], self.state_size[1]))
        next_states = np.zeros((self.batch_size, self.state_size[0], self.state_size[1]))
        actions, rewards, is_done = [], [], []
        for i in range(self.batch_size):
            states[i] = minibatch[i][0]
            actions.append(minibatch[i][1])
            rewards.append(minibatch[i][2])
            next_states[i] = minibatch[i][3]
            is_done.append(minibatch[i][4])

        targets = self.model.predict(states, verbose=0)
        targets_next = self.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if is_done[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(targets_next[i])

        # Train the Neural Network with batches where target is the value function
        self.model.fit(states, targets, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def train(self):
        rewards = []
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            done = False
            steps_survived = 0
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # if terminated:
                #     reward = -100

                steps_survived += 1
                total_reward += reward

                # Add this SARS to memory
                self.remember(state, action, reward, next_state, done)
                state = next_state

                # Experience replay
                self.replay()

                if done:
                    dateTimeObj = datetime.now()
                    timestampStr = dateTimeObj.strftime("%H:%M:%S")
                    print('episode: {}/{}, steps survived: {}, total reward: {}, e: {:.2}, time: {}'.format(e + 1,
                                                                                                            self.EPISODES,
                                                                                                            steps_survived,
                                                                                                            total_reward,
                                                                                                            self.epsilon,
                                                                                                            timestampStr))

            rewards.append(total_reward)
        print('Saving trained model as highway-dqn-train.h5')
        self.save("./save/highway-dqn-train.h5")
        return rewards

    # test function if you want to test the learned model
    def test(self):
        self.load("./save/highway-dqn-train.h5")
        for e in range(self.EPISODES):
            state, _ = self.env.reset()
            state = state.reshape([1, self.state_size[0], self.state_size[1]])
            done = False
            steps_survived = 0
            total_reward = 0
            while not done:
                action = np.argmax(self.model.predict(state, verbose=0))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state.reshape([1, self.state_size[0], self.state_size[1]])
                steps_survived += 1
                total_reward += reward
                self.env.render()
                if done:
                    print('episode: {}/{}, steps survived: {}, total reward: {}'.format(e + 1,
                                                                                        self.EPISODES,
                                                                                        steps_survived,
                                                                                        total_reward))


def main():
    agent = DQNAgent()
    rewards = agent.train()
    with open(f'dqn_rewards.txt', 'w') as f:
        f.write(str(rewards))
    plot_df = pd.DataFrame({'episodes': list(range(1, len(rewards) + 1)), 'timesteps_survived': rewards})
    sns.lineplot(plot_df, x='episodes', y='timesteps_survived')
    agent = DQNAgent()
    agent.test()


if __name__ == '__main__':
    main()
