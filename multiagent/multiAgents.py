# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import math
from pickle import FALSE
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    #This algorithm is based on the of the minimax search algorithm on page 8 of lecture slide 5
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        #Calculates max-value for the player aka. agent 0
        def maxValue(gameState,depth, returnAction):
            if gameState.isWin() or gameState.isLose() or depth==self.depth: #checks for terminal state
                return self.evaluationFunction(gameState)
            bestValue = -math.inf
            bestAction = '' 
            for action in gameState.getLegalActions(0): #tries to find the best action and its respective value amongs the action set
                successor= gameState.generateSuccessor(0,action)
                newValue = minValue(successor,depth,1, False)
                if newValue > bestValue:
                    bestAction, bestValue = action, newValue
            return bestAction if returnAction else bestValue #returns either the best action or its respective value depending on returnAction boolean
        
        #Calculates min-value for the ghosts aka. agents 1, 2 etc.
        def minValue(gameState,depth, agentIndex, returnAction): 
            if gameState.isWin() or gameState.isLose(): #checks for terminal state 
                return self.evaluationFunction(gameState)
            bestValue = math.inf
            bestAction = ''
            for action in gameState.getLegalActions(agentIndex): #tries to find the best action and its respective value amongs the action set
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents() - 1): #checks if the next agent is the player
                    newValue = maxValue(successor,depth+1, False)
                else: #next agent is another ghost
                    newValue = minValue(successor,depth,agentIndex+1, False)
                if newValue < bestValue:
                    bestAction, bestValue = action,newValue
            return bestAction if returnAction else bestValue #returns either the best action or its respective value depending on returnAction boolean
        
        return maxValue(gameState,0,True) #return best action, player starts so agentIndex=0

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    #This algorithm is based on the of the alpha-beta search algorithm on page 21 of lecture slide 5
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        #Calculates max-value for the player aka. agent 0
        def maxValue(gameState,depth,alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth ==self.depth: #checks for terminal state
                return self.evaluationFunction(gameState)
            value = -math.inf
            alpha1 = alpha
            for action in gameState.getLegalActions(0): #tries to find the best value amongs the action set
                successor= gameState.generateSuccessor(0,action)
                value = max (value,minValue(successor,depth,1,alpha1,beta))
                if value > beta:
                    return value
                alpha1 = max(alpha1,value)
            return value
        
        #Calculates min-value for the ghosts aka. agents 1, 2 etc.
        def minValue(gameState,depth,agentIndex,alpha,beta):
            if gameState.isWin() or gameState.isLose(): #checks for terminal state
                return self.evaluationFunction(gameState)
            value = math.inf
            beta1 = beta
            for action in gameState.getLegalActions(agentIndex): #tries to find the best value amongs the action set
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents()-1): #checks if the next agent is the player
                    value = min (value,maxValue(successor,depth+1,alpha,beta1))
                else: #next agent is another ghost
                    value = min(value,minValue(successor,depth,agentIndex+1,alpha,beta1))
                if value < alpha:
                    return value
                beta1 = min(beta1,value)
            return value

        #basicly the same as maxValue but returns action and not value
        currentScore = -math.inf
        alpha, beta = -math.inf, math.inf
        returnAction = ''
        for action in gameState.getLegalActions(0): #returns best action to take
            successor = gameState.generateSuccessor(0,action)
            score = minValue(successor,0,1,alpha,beta)
            if score > currentScore:
                returnAction, currentScore = action, score
            if score > beta:
                return returnAction
            alpha = max(alpha,score)
        return returnAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
