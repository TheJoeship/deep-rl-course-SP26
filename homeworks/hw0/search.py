# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #test commands to understand:
    #print("Start:", problem.getStartState()) #(5,5)
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState())) #False ofc since we havent moved
    #print("Start's successors:", problem.getSuccessors(problem.getStartState())) #list of triples of possible moves i.e (5,4) south 1
    #DFS:
    stack = util.Stack()
    visited = []

    stack.push([problem.getStartState(),[]]) #current coordinate + path to that coordinate []

    while not stack.isEmpty():
        (node,path) = stack.pop() #node should be of form (x,y)
        #print(node)
        #print(path)
        if node in visited:
            continue
        if problem.isGoalState(node):
                path.reverse() #reverse list for proper navigation order
                return path
        visited.append(node)   

        #traverse all edges
        successors = problem.getSuccessors(node)
        size = len(successors)
        for i in range(size-1,-1,-1):
            v = successors[i]
            v_node = v[0]
            if v_node not in visited:
                stack.push([v_node,[v[1]]+path])

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    queue = util.Queue()
    visited = []

    queue.push([problem.getStartState(),[]]) #[current coordinate, path to current]

    while not queue.isEmpty():
        (node,path) = queue.pop() #node should be of form (x,y)
        #print(node)
        #print(path)
        if node in visited:
            continue
        if problem.isGoalState(node):
                path.reverse() #reverse list for proper navigation order
                return path
        visited.append(node)   

        #traverse all edges
        successors = problem.getSuccessors(node)
        size = len(successors)
        for i in range(size-1,-1,-1):
            v = successors[i]
            v_node = v[0]
            if v_node not in visited:
                queue.push([v_node,[v[1]]+path])
    

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    pqueue = util.PriorityQueue()
    visited = []
    #tracking TOTAL path cost not just cost of each move so we need to keep track of 3 things now
    pqueue.push([problem.getStartState(),[],0],0) #[current,path to current,cost of path] + priority

    while not pqueue.isEmpty():
        (node,path,path_cost) = pqueue.pop() #node should be of form (x,y)
        if node in visited:
            continue
        if problem.isGoalState(node):
                path.reverse() #reverse list for proper navigation order
                return path
        visited.append(node)   

        #traverse all edges
        successors = problem.getSuccessors(node)
        size = len(successors)
        for i in range(size-1,-1,-1):
            v = successors[i]
            v_node = v[0]
            cost = v[2] #priority by cost
            if v_node not in visited:
                pqueue.update([v_node,[v[1]]+path,cost+path_cost],cost+path_cost)

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    pqueue = util.PriorityQueue()
    best_cost = {problem.getStartState(): 0} #Need to swap visited for best cost to avoid popping nodes inadvertantley 
    start_h = heuristic(problem.getStartState(),problem)
    pqueue.push([problem.getStartState(),[],0],start_h) #[current,path to current,cost of path] + priority

    while not pqueue.isEmpty():
        #path_cost is cost from start to current node
        (node,path,path_cost) = pqueue.pop()

        if path_cost > best_cost[node]: #check if we prev found better path
            continue
        if problem.isGoalState(node):
                path.reverse() #reverse list for proper navigation order
                return path 
        #traverse all edges
        successors = problem.getSuccessors(node)
        size = len(successors)
        for i in range(size-1,-1,-1):
            v = successors[i]
            v_node = v[0]
            cost = v[2] #extract cost of moving to node i
            new_cost = cost+path_cost #g(n)
            priority = new_cost + heuristic(v_node,problem) #priority = new_cost + heuristic(n) (f(n) = g(n) + h(n))
            if v_node not in best_cost or new_cost < best_cost[v_node]: #replaces visited functionality
                best_cost[v_node] = new_cost
                pqueue.push([v_node,[v[1]]+path,new_cost],priority)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
