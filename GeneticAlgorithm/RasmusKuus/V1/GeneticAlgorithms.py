from random import random, randint
import numpy as np

class GeneticAlgorithmsClass:
    def __init__(self,inputs,layers,neurons,outputs,activation,pops):
        self.inputs = inputs + 1 # bias
        self.neurons = neurons + 1 # bias
        self.layers = layers
        self.outputs = outputs
        self.activation = activation

        self.selection = 0.10 # Selected for breeding
        self.crossOverRate = 0.75 # Chance for crossover between genes
        self.mutationRate = 1 # 1/pops # Chance for mutation
        self.elitisme = 0.10
        
        self.HiddenLayersOutput = np.array([])

        self.createOutput()
        self.createWeigths()

        self.pops = pops
        self.population = np.array([])
        self.populationScores = np.zeros(self.pops)

        self.createPopulation()
        self.generation = 0

    def createOutput(self):
        # outputs
        tempList = []
        outputs = []
        for _ in range(self.inputs):
            tempList.append(0.0)

        outputs.append(np.array(tempList.copy()))
        tempList.clear()

        for _ in range(self.layers):
            for _ in range(self.neurons):
                tempList.append(0.0)
            outputs.append(np.array(tempList.copy()))
            tempList.clear()

        for _ in range(self.outputs):
            tempList.append(0.0)

        outputs.append(np.array(tempList.copy()))
        tempList.clear()
        self.HiddenLayersOutput = np.array(outputs.copy())

    def createWeigths(self):
        # weights
        tempList = []
        weights = []
        tempInnerList = []
        sigma = 0.01
        # Each neuron has a list of weights for the inputs
        if self.layers != 0:
            for _ in range(self.neurons):
                for _ in range(self.inputs):
                    tempInnerList.append(sigma * np.random.randn(1)[0])
                tempList.append(np.array(tempInnerList.copy()))
                tempInnerList.clear()

            weights.append(np.array(tempList.copy()))
            tempList.clear()

            for _ in range(self.layers-1):
                for _ in range(self.neurons):
                    for _ in range(self.neurons):
                        tempInnerList.append(sigma * np.random.randn(1)[0])
                    tempList.append(np.array(tempInnerList.copy()))
                    tempInnerList.clear()
                weights.append(np.array(tempList.copy()))
                tempList.clear()
            
            # Each output has a list of weigths for the neurons in the last layer
            for _ in range(self.outputs):
                for _ in range(self.neurons):
                    tempInnerList.append(sigma * np.random.randn(1)[0])
                tempList.append(np.array(tempInnerList.copy()))
                tempInnerList.clear()
            #print(self.layers)
            #print(len(tempList))
            weights.append(np.array(tempList.copy()))
        else:
            for _ in range(self.outputs):
                for _ in range(self.inputs):
                    tempInnerList.append(sigma * np.random.randn(1)[0])
                tempList.append(np.array(tempInnerList.copy()))
                tempInnerList.clear()
            #print(self.layers)
            #print(len(tempList))
            weights.append(np.array(tempList.copy()))

        return np.array(weights.copy())

    def createPopulation(self):
        temp = []
        for _ in range(self.pops):
            temp.append(np.array(self.createWeigths()))
        self.population = np.array(temp)

    def getPopulation(self,idx):
        return self.population[idx]

    def getScore(self,idx):
        return self.populationScores[idx]

    def setPopulationScore(self,idx,score):
        self.populationScores[idx] = score

    def randomMutation(self,HiddenLayersWeigth):
        mutant = HiddenLayersWeigth.copy()
        sigma = 0.001
        for i in range(self.layers):
            for j in range(self.neurons):
                for k in range(len(self.HiddenLayersOutput[i])):
                    if random() < self.mutationRate:
                        mutant[i][j][k] += (sigma * np.random.randn(1)[0])
        
        for i in range(self.outputs):
            for j in range(len(self.HiddenLayersOutput[self.layers])):
                if random() < self.mutationRate:
                    mutant[self.layers][i][j] += (sigma * np.random.randn(1)[0])

        return mutant

    def twoPointCrossOver(self,HiddenLayersWeigth1, HiddenLayersWeigth2):
        mutant = [HiddenLayersWeigth1.copy(),HiddenLayersWeigth2.copy()]

        for i in range(self.layers):
            for j in range(self.neurons):
                for k in range(len(self.HiddenLayersOutput[i])):
                    if random() < self.crossOverRate:
                        mutant[0][i][j][k], mutant[1][i][j][k] = mutant[1][i][j][k], mutant[0][i][j][k]
        
        for i in range(self.outputs):
            for j in range(len(self.HiddenLayersOutput[self.layers])):
                if random() < self.crossOverRate:
                    mutant[0][self.layers][i][j], mutant[1][self.layers][i][j] = mutant[1][self.layers][i][j], mutant[0][self.layers][i][j]

        return mutant

    def singlePointCrossOver(self,HiddenLayersWeigth1, HiddenLayersWeigth2):
        mutant = [HiddenLayersWeigth1.copy(),HiddenLayersWeigth2.copy()]

        mutateWho = False

        for i in range(self.layers):
            for j in range(self.neurons):
                for k in range(len(self.HiddenLayersOutput[i])):
                    if random() < self.crossOverRate:
                        if mutateWho:
                            mutateWho = False
                            mutant[0][i][j][k] = mutant[1][i][j][k]
                        else:
                            mutateWho = True
                            mutant[1][i][j][k] = mutant[0][i][j][k]
        
        for i in range(self.outputs):
            for j in range(len(self.HiddenLayersOutput[self.layers])):
                if random() < self.crossOverRate:
                    if mutateWho:
                        mutateWho = False
                        mutant[0][self.layers][i][j] = mutant[1][self.layers][i][j]
                    else:
                        mutateWho = True
                        mutant[1][self.layers][i][j] = mutant[0][self.layers][i][j]

        return mutant

    def uniformPointCrossOver(self,HiddenLayersWeigth1, HiddenLayersWeigth2):
        mutant = [HiddenLayersWeigth1.copy(),HiddenLayersWeigth2.copy()]

        for i in range(self.layers):
            for j in range(self.neurons):
                for k in range(len(self.HiddenLayersOutput[i])):
                    if random() < self.crossOverRate:
                        if bool(randint(0,1)):
                            mutant[0][i][j][k] = mutant[1][i][j][k]
                        else:
                            mutant[1][i][j][k] = mutant[0][i][j][k]
        
        for i in range(self.outputs):
            for j in range(len(self.HiddenLayersOutput[self.layers])):
                if random() < self.crossOverRate:
                    if bool(randint(0,1)):
                        mutant[0][self.layers][i][j] = mutant[1][self.layers][i][j]
                    else:
                        mutant[1][self.layers][i][j] = mutant[0][self.layers][i][j]

        return mutant

    def mutate(self):
        # Roulette wheel
        idx = np.random.choice(self.pops,int(self.pops*self.selection),replace=False,p=np.divide(self.populationScores,np.sum(self.populationScores)))
        tempList = []
        for i in range(len(idx)):
            tempList.append(self.population[idx[i]].copy())
        
        for counter in range(self.pops):
            if counter%int(self.pops*self.selection) == 0:
                np.random.shuffle(tempList)
            self.population[i] = tempList[counter%len(tempList)]

        for counter in range((self.pops)-int(self.pops*self.elitisme)):
            if counter%2 == 0:
                mutants = self.singlePointCrossOver(self.population[counter], self.population[counter+1])
                self.population[counter], self.population[counter+1] = mutants[0],mutants[1]
            
        # random mutations
        for counter in range(self.pops):
            self.population[counter] = self.randomMutation(self.population[counter]).copy()
        
        self.populationScores = np.zeros(self.pops)

    def runModel(self,newInput,HiddenLayersWeigth):
        self.HiddenLayersOutput[0] = np.append(newInput,1)

        for i in range(self.layers):
            #if i != 0:
            #    self.HiddenLayersOutput[i] = self.HiddenLayersOutput[i]/np.sum(self.HiddenLayersOutput[i])
            for j in range(self.neurons):
                if j == 0:
                    self.HiddenLayersOutput[i+1][j] = 1
                else:
                    self.HiddenLayersOutput[i+1][j] = self.activation(HiddenLayersWeigth[i][j] @ self.HiddenLayersOutput[i])

        outputs = np.zeros(self.outputs)
        for i in range(self.outputs):
            outputs[i] = HiddenLayersWeigth[self.layers][i] @ self.HiddenLayersOutput[self.layers]

        return outputs

    def exportWeigths(self, i,wins):
        np.save("Gen-"+str(self.generation-1)+"-Wins-"+str(wins)+"-"+str(self.inputs-1)+"-"+str(self.layers)+"-"+str(self.neurons-1)+"-"+str(self.outputs)+"-weigths.npy", self.population[i])

    def importWeigths(self,i,name):
        self.population[i] = np.load(name, allow_pickle=True)

    def exportPopulation(self):
        np.save("Gen-"+str(self.generation)+"-"+str(self.inputs-1)+"-"+str(self.layers)+"-"+str(self.neurons-1)+"-"+str(self.outputs)+"-weigths.npy", self.population)
        self.generation += 1

    def importPopulation(self, name):
        self.population = np.load(name, allow_pickle=True)