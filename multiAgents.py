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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        score = successorGameState.getScore()
        #find the closestghost
        closestGhost = float("inf")
        for ghost in newGhostStates:
            ghostpos = ghost.getPosition()
            distance = manhattanDistance(newPos, ghostpos)
            if(closestGhost > distance):
                closestGhost = distance
                if closestGhost == 0:
                    closestGhost = 1
        #find the average food distance and closet food
        foodList = newFood.asList()
        avgFoodDistance = 1
        closestFood = float("inf")
        count = 0
        foodDistance = 0
        for food in foodList:
            distance = manhattanDistance(food,newPos)
            count += 1
            foodDistance += distance
            if closestFood > distance:
                closestFood = distance
        if count != 0:
            avgFoodDistance = foodDistance/count

        #print "closestfood:", closestFood, "avgFoodDistance:", avgFoodDistance, "closestGhost", closestGhost, "score:", score
            #if(closetGhost > closestFood)
            #score += 10
	
        if(closestGhost > 2)or(newScaredTimes[0] >= 1):    #movement regarding to scared ghost(maybe optimize this later)
            if(action == Directions.STOP):
		score -= 0
	    return (score + 10/closestFood)
        else:
            return (score + closestGhost + 20/avgFoodDistance)

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
        """
        "*** YOUR CODE HERE ***"
        #actionlist = []
        def minimaxDecision(state, depth,index):
            if(index > gameState.getNumAgents()-1):
                index = 0
                depth -= 1
            if state.isLose() or state.isWin() or depth == 0:
                #print "reach to end", self.evaluationFunction(state)
                return (self.evaluationFunction(state), Directions.STOP)
            if(index == 0): #index = 0 means its pacman, otherwise its ghost
                return maxValue(state, depth, index)
            else:
                return minValue(state, depth,index)
                
                
            #max for pacman
        def maxValue(state, depth, index):
            legalActions = state.getLegalActions(index)
            actionlist = [(minimaxDecision(state.generateSuccessor(index, action),depth,index+1)[0], action) for action in legalActions]
            return max(actionlist)
        
            #min for ghost
        def minValue(state, depth, index):
            legalActions = state.getLegalActions(index)
            actionlist = [(minimaxDecision(state.generateSuccessor(index, action),depth,index+1)[0], action) for action in legalActions]
            return min(actionlist)
        
        return minimaxDecision(gameState,self.depth,0)[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state, depth, index, alpha, beta):
            if(index > gameState.getNumAgents()-1):
                index = 0
                depth -= 1
            if state.isLose() or state.isWin() or depth == 0:
                return (self.evaluationFunction(state), Directions.STOP)
            if(index == 0):
                return maxValue(state, depth, index, alpha, beta)
            else:
                return minValue(state, depth, index, alpha, beta)


        #max for pacman
        def maxValue(state, depth, index, alpha, beta):
            v = float('-inf')
            move = Directions.STOP
            legalActions = state.getLegalActions(index)
            #original way doesnt work cuz they visited extra states
            """
            "actionlist = [(value(state.generateSuccessor(index, action),depth,index+1,alpha,beta)[0], action) for action in legalActions]

            "next = max(actionlist)
            "if(next[0] > beta):
            "    return next
            "alpha = max(alpha, next[0])
            "return next
            """
            for action in legalActions:
                next = (value(state.generateSuccessor(index, action), depth, index+1, alpha, beta)[0],action)
                #v = max(v, next)
                if v < next[0]:
                    v = next[0]
                    move = action
                nextState = (v, move)
                if v > beta:
                    return nextState
                alpha = max(alpha, v)
            return nextState
        
            
        #min for ghost
        def minValue(state, depth, index, alpha, beta):
            v = float('inf')
            legalActions = state.getLegalActions(index)
            
            for action in legalActions:
                next = (value(state.generateSuccessor(index, action), depth, index+1, alpha, beta)[0],action)
                #v = min(v, next)
                if v > next[0]:
                    v = next[0]
                    move = action
                nextState = (v, move)
                if v < alpha:
                    return nextState
                beta = min(beta, v)
            return nextState

        return value(gameState,self.depth,0,float('-inf'), float('inf'))[1]
   


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
        #actionlist = []
        def exDecision(state, depth,index):
            if(index > gameState.getNumAgents()-1):
                index = 0
                depth -= 1
            if state.isLose() or state.isWin() or depth == 0:
                #print "reach to end", self.evaluationFunction(state)
                return (self.evaluationFunction(state), Directions.STOP)
            if(index == 0): #index = 0 means its pacman, otherwise its ghost
                return maxValue(state, depth, index)
            else:
                return exValue(state, depth,index)

        #max for pacman
        def maxValue(state, depth, index):
            legalActions = state.getLegalActions(index)
            actionlist = [(exDecision(state.generateSuccessor(index, action),depth,index+1)[0], action) for action in legalActions]
            return max(actionlist)
        
        #ex for ghost  only calculated the probability here.
        def exValue(state, depth, index):
            score = 0
            move = Directions.STOP
            legalActions = state.getLegalActions(index)
            length = len(legalActions)
            for action in legalActions:
                score += exDecision(state.generateSuccessor(index, action),depth,index+1)[0]
            return (float(score)/length, Directions.STOP)
        
        return exDecision(gameState,self.depth,0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    wall = currentGameState.getWalls()
    walllist = wall.asList()
    foodList = newFood.asList()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()
    #DFS
    frontier = util.Stack()
    explored = set()
    cur = newPos
    hashtable = {cur : 0}
    frontier.push(cur)
    #print "cur is ", cur
    from game import Actions
    """
    while not frontier.isEmpty():
    	cur = frontier.pop()
    	explored.add(cur)
        if cur in foodList:
    		break
        for next in getSuccessorPos(cur):
    		if not nextPoint in explored:
    			frontier.push(next)
    			hashtable[next] = hashtable[cur] + 1    

    print "DFS is", hashtable[cur]
    """
    """
    def getSuccessorPos(pos):
    	successor = []
    	successor.append(pos)
    	for direction in [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]:
    		while True:
			curPos = successor[-1]
    			dx, dy = Actions.directionToVector(direction)
    			nextPos = (curPos[0] + dx, curPos[1] + dy)
			if nextPos not in walllist:
				print"now append", nextPos, "direction moving", direction
    				successor.append(nextPos)
    			else:
    				break
    	return successor
    successor = getSuccessorPos(cur)
    print"successor", successor[1],succesor[2]
    """
    #find the closestghost
    closestGhost = float("inf")
    for ghost in newGhostStates:
        ghostpos = ghost.getPosition()
        distance = manhattanDistance(newPos, ghostpos)
        if(closestGhost > distance):
            closestGhost = distance
            if closestGhost == 0:
                closestGhost = 1
    #find the average food distance and closet food
    
    avgFoodDistance = 1
    closestFood = float("inf")
    count = 0
    foodDistance = 0
    closestFoodPos = (0,0)
    for food in foodList:
        distance = manhattanDistance(food,newPos)
        count += 1
        foodDistance += distance
        if closestFood > distance:
            closestFood = distance
            closestFoodPos = food    	
    if count != 0:
        avgFoodDistance = foodDistance/count
    
    print "nearest food is at", closestFoodPos, "cur is", newPos    
    low = 0, 
    high = 0
    passwalls = 0
    if(cur[0] == closestFoodPos[0]):
    	if cur[1] > closestFoodPos[1]:
    		low = closestFoodPos[1]
    		high = cur[1]
    	else:
    		low = cur[1]
    		high = closestFoodPos[1]
    	for i in range(low,high):
    		if(cur[0],i) in walllist:
    			print"reduce score"
    			passwalls = -100
    elif(cur[1] == closestFoodPos[1]):
    	if cur[0] > closestFoodPos[0]:
    		low = closestFoodPos[0]
    		high = cur[0]
    	else:
    		low = cur[0]
    		high = closestFoodPos[0]
    	for i in range(low,high):
    		if(i,cur[1]) in walllist:
    			print"reduce score"
    			passwalls = -5
	    
    #print "closestfood:", closestFood, "avgFoodDistance:", avgFoodDistance, "closestGhost", closestGhost, "score:", score
    #if(closetGhost > closestFood)
    #score += 10
    
    if(closestGhost > 2)or(newScaredTimes[0] >= 1):    #movement regarding to scared ghost(maybe optimize this later)
        print "closest food", 20/closestFood, "score", currentGameState.getScore(), "passwalls", passwalls
    	print "final is ", (score + 7/closestFood + passwalls)
	return (score + 7/closestFood + passwalls)
    else:
        return (score + closestGhost + 20/avgFoodDistance)

# Abbreviation
better = betterEvaluationFunction


