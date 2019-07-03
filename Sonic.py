import retro
import numpy as np
import cv2
import neat
import pickle
# py 3.6
env = retro.make('SuperMarioWorld-Snes', 'Bridges1')  # openai搭建环境

imgarray = []
resume = False
restore_file = ""


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        observation = env.reset()
        inx, iny, inc = env.observation_space.shape  # x, y color
        inx = int(inx / 8)
        iny = int(iny / 8)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)  # 创建循环神经网络

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0

        xpos_max = 0

        done = False

        while not done:   # 当个体没有结束训练时
            env.render()
            frame += 1
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            imgarray = np.ndarray.flatten(observation)
            nn_output = net.activate(imgarray)
            observation, reward, done, info = env.step(nn_output)

            xpos_end = info['screen_x_end'] # 获取终点位置
            xpos = info['x']  # 角色当前位置

            if xpos == xpos_end and xpos > 500:
                fitness_current += 100000    # 到达终点就训练结束
                done = True

            if xpos > xpos_max:  # 往右移动就加适应度
                fitness_current += 1
                xpos_max = xpos

            fitness_current += reward

            if fitness_current > current_max_fitness: # 适应度上升就不会重开
                current_max_fitness = fitness_current
                counter = 0

            else:
                counter += 1

            if done or counter > 250:    # 这时候要结束这个个体，counter为250时大约3s
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

if resume == True:  # 是否继续上次检查点的训练
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)    # 重新训练

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)  # 最终会得出最优的模型

with open('winner.pkl', 'wb') as output:   # 训练完成，保存模型
    pickle.dump(winner, output, 1)
