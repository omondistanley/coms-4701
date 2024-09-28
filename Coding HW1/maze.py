__author__ = "Stanley Omondi"
__email__ = "soo2117@columbia.edu"

#======================================================================#
#*#*#*# Optional: Import any allowed libraries you may need here #*#*#*#
#======================================================================#
import queue
import heapq
import time
import resource
#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Robot Path Planning | HW 1 | COMS 4701')
	parser.add_argument('-bfs', action="store_true", default=False , help="Run BFS on the map")
	parser.add_argument('-dfs', action="store_true", default=False, help= "Run DFS on the map")
	parser.add_argument('-astar', action="store_true", default=False, help="Run A* on the map")
	parser.add_argument('-ida', action="store_true", default=False, help="Run Iterative Deepening A* on the map")
	parser.add_argument('-all', action="store_true", default=False, help="Run all the 4 algorithms")
	parser.add_argument('-m', action="store", help="Map filename")

	results = parser.parse_args()

	if results.m=="" or not(results.all or results.astar or results.bfs or results.dfs or results.ida):
		print("Check the parameters : >> python hw1_UNI.py -h")
		exit()

	if results.all:
		results.bfs = results.dfs = results.astar = results.ida = True

	# Reading of map given and all other initializations
	try:
		with open(results.m) as f:
			arena = f.read()
			arena = arena.split("\n")
	except:
		print("Error in reading the arena file.")
		exit()

	# Internal representation
	print(arena)

	print("The arena of size "+ str(len(arena)) + "x" + str(len(arena[0])))
	print("\n".join(arena))

class MazeState:
	'''
	This class is an abstraction to store a maze state, which contains the following:
	- Maze configuration (arena)
	- Current Position (position in the the maze that the current state represents)
	- Parent (the state from which the current state came from)
	- Action (the action taken in the parent state, direction moved, which lead to the creation of the current state)
	- Cost (Cost  of the path taken from the start to the current state)
	- Children (a child of the current state is generated by moving in a direction)
	'''
	
	def get_start_index(self):
		'''
		Returns the start index of the maze based on the given arena
		returns (-1, -1) if no start index found
		'''
		#=======================================================================#
		#*#*#*# TODO: Write your code to find the start index of the maze #*#*#*#
		#=======================================================================#
		maze = self.arena
		start = (-1, -1)
		for i in range(len(maze)):
			for j in range(len(maze[0])):
				if maze[i][j] == "s":
					start = (i, j)
					#print(start)
					return start
		#print(start)
		return start
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#

	def get_goal_index(self):
		'''
		Returns the goal index of the maze based on the given arena
		returns (-1, -1) if no goal index found
		'''
		#======================================================================#
		#*#*#*# TODO: Write your code to find the goal index of the maze #*#*#*#
		#======================================================================#
		maze = self.arena
		goal = (-1, -1)
		for i in range(len(maze)):
			for j in range(len(maze[0])):
				if maze[i][j] == "g":
					goal = (i, j)
					#print(goal)
					return goal	
		#print(goal)
		return goal
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#

	def __init__(self, arena, parent=None, action='Start', cost=0, current_position=(-1,-1)):

		self.arena = arena
		self.parent = parent
		self.action = action
		self.cost = cost
		self.children = []

		self.start = self.get_start_index()
		self.goal = self.get_goal_index()
		if(current_position[0] == -1):
			self.current_position = self.start
		else:
			self.current_position = current_position

	def display(self):
		print("\n".join(self.arena))

	def move_up(self):
		'''
		This function checks if up is a valid move from the given state.
		If up is a valid move, returns a child in which the player has moved up
		Else returns None.
		'''

		#=================================================================#
		#*#*#*# TODO: Write your code to move up in the puzzle here #*#*#*#
		#=================================================================#
		maze = self.arena
		pos = self.current_position
		#child = self.children
		i = pos[0]
		j = pos[1]
		if i > 0 and maze[i - 1][j] != "o":
			pos = (i -1, j)
			
			child_up = MazeState (
				arena = self.arena,
				parent = self,
				action='up',
				cost = self.cost + 1, 
				current_position=pos
			)
			return child_up

		return None
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#


	def move_down(self):
		'''
		This function checks if down is a valid move from the given state.
		If down is a valid move, returns a child in which the player has moved down.
		Else returns None.
		'''
		
		#===================================================================#
		#*#*#*# TODO: Write your code to move down in the puzzle here #*#*#*#
		#===================================================================#
		maze = self.arena
		pos = self.current_position
		i = pos[0]
		j = pos[1]
		if i + 1 < len(maze) and maze[i + 1][j] != "o":
			pos = (i + 1, j)

			child_down = MazeState (
				arena = self.arena,
				parent = self,
				action='down',
				cost = self.cost+1,
				current_position=pos
			)
			return child_down

		return None
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#

	def move_left(self):
		'''
		This function checks if left is a valid move from the given state.
		If left is a valid move, returns a child in which the player has moved left.
		Else returns None.
		'''
		
		#===================================================================#
		#*#*#*# TODO: Write your code to move left in the puzzle here #*#*#*#
		#===================================================================#
		maze = self.arena
		pos = self.current_position
		i = pos[0]
		j = pos[1]
		if j > 0 and maze[i][ j -1] != "o":
			pos = (i, j -1)
			child_left = MazeState (
				arena = self.arena,
				parent = self,
				action='left',
				cost = self.cost+1,
				current_position=pos
			)
			return child_left
		
		return None
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#


	def move_right(self):
		'''
		This function checks if left is a valid move from the given state.
		If left is a valid move, returns a child in which the player has moved left.
		Else returns None.
		'''
		
		#====================================================================#
		#*#*#*# TODO: Write your code to move right in the puzzle here #*#*#*#
		#====================================================================#
		maze = self.arena
		pos = self.current_position
		i = pos[0]
		j = pos[1]
		if j + 1 < len(maze[i]) and maze[i][j + 1] != "o":
			pos = (i, j+1)
	
			child_right = MazeState (
				arena = self.arena,
				parent = self,
				action='right',
				cost = self.cost+1,
				current_position=pos
			)
			return child_right
	
		return None
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#

	def expand(self):
		""" 
		Generate the child nodes of this node 
		"""
		
		if(len(self.children) != 0):
			return self.children

		# Do not change the order in this function, since the grading script assumes this order of expansion when checking
		children = [self.move_up(), self.move_right(), self.move_down(), self.move_left()]

		self.children = [state for state in children if state is not None]
		#print(len(self.children))
		return self.children
		
	def __hash__(self):
		'''
		Maze states hashed based on cost. 
		This function may be modified if required.
		'''
		#============================================================================================#
		#*#*#*# Optional: May be modified if your algorithm requires a different hash function #*#*#*#
		#============================================================================================#
		
		return self.cost
		
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#
	
		
	def __eq__(self, other):
		'''
		Maze states are defined as equal if they have the same dimensions and the same current position. 
		This function may be modified if required.
		'''
		
		#=============================================================================================#
		#*#*#*# Optional: May be modified if your algorithm requires a different equality check #*#*#*#
		#=============================================================================================#
		
		m1 = self.arena
		m2 = other.arena

		if(len(m1) != len(m2)):
			return False

		for i in range(0, len(m1)):
			if(not (m1[i] == m2[i])):
				return False
		return self.current_position == other.current_position
		
		#=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#
		
	#=====================================================================================#
	#*#*#*# Optional: Write any other functions you may need in the MazeState Class #*#*#*#
	#=====================================================================================#
		
	#=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#



#================================================================================#
#*#*#*# Optional: You may write helper functions in this space if required #*#*#*#
#================================================================================#
	
#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#


'''
This function runs Breadth First Search on the input arena (which is a list of str)
Returns a ([], int) tuple where the [] represents the solved arena as a list of str and the int represents the cost of the solution
'''
def bfs(arena):
	#=================================================#
	#*#*#*# TODO: Write your BFS algorithm here #*#*#*#
	#=================================================#
	nodesexpanded = -1
	startram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	starttime = time.time()

	currstate = MazeState(arena)
	arenastart = currstate.start
	arenagoal = currstate.goal
	frontier = queue.Queue()
	explored = set()
	frontier.put(currstate)
	path = []
	parent = {currstate: None}
	max_nodes_stored = 1

	while not frontier.empty():
		current = frontier.get()
		explored.add(current.current_position)
		
		if current.current_position == arenagoal:
			cost = current.cost
			pathToGoal = []
			while current:
				pathToGoal.append(arena[current.current_position[0]][current.current_position[1]])
				path.append(current)
				current = parent[current]
			pathToGoal.reverse()
			goalDepth = len(pathToGoal)
			endram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			endtime = time.time()
			maxram = endram - startram
			runtime = endtime - starttime
			return pathToGoal, cost, nodesexpanded, -1, -1, runtime, maxram

		for valid_move in current.expand():
			if valid_move and valid_move.current_position not in explored:
				frontier.put(valid_move)
				explored.add(valid_move.current_position)
				parent[valid_move] = current
	
	endram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	endtime = time.time()
	runtime = endtime - starttime
	maxram = endram - startram
	return [], -1, nodesexpanded, -1, -1, runtime, maxram # Replace with return values
	#=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#

'''
This function runs Depth First Search on the input arena (which is a list of str)
Returns a ([], int) tuple where the [] represents the solved arena as a list of str and the int represents the cost of the solution
'''
def dfs(arena):

	#=================================================#
	#*#*#*# TODO: Write your DFS algorithm here #*#*#*#
	#=================================================#
	startram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	starttime = time.time()

	currstate = MazeState(arena)
	arenastart = currstate.start
	arenagoal = currstate.goal
	#print(arenastart)
	#print(arenagoal)
	frontier = []
	explored = set()
	frontier.append(currstate)
	path = []
	parent = {currstate: None}
	max_nodes_stored = 1

	while frontier:
		current = frontier.pop()
		explored.add(current.current_position)

		if current.current_position == arenagoal:
			cost = current.cost
			dfs_path = []
			while current:
				dfs_path.append(arena[current.current_position[0]][current.current_position[1]])
				path.append(current)
				current = parent[current]
			endram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			goalDepth = len(dfs_path)
			endtime = time.time()
			maxram = endram - startram
			runtime = endtime - starttime
			return dfs_path, cost, -1, -1, -1, runtime, maxram
		
		moves = current.expand()
		for valid_move in reversed(moves):
			if valid_move not in frontier and valid_move.current_position not in explored:
				frontier.append(valid_move)
				explored.add(valid_move.current_position)
				parent[valid_move] = current
	endram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	endtime = time.time()
	runtime = endtime - starttime
	maxram = endram - startram
	return [], -1, -1, -1, -1, runtime, maxram # Replace with return values
	#=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#

'''
This function runs A* Search on the input arena (which is a list of str)
Returns a ([], int) tuple where the [] represents the solved arena as a list of str and the int represents the cost of the solution
'''
def astar(arena):

	#================================================#
	#*#*#*# TODO: Write your A* algorithm here #*#*#*#
	#================================================#
	startram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	starttime = time.time()

	currstate = MazeState(arena)
	arenastart = currstate.start
	arenagoal = currstate.goal

	frontier = queue.PriorityQueue()
	explored = set()
	frontier.put((0, currstate))
	parent = {currstate: None}
	path =[]

	while not frontier.empty():
		priority, current = frontier.get()
		explored.add(current.current_position)
		#path.append(current)

		if current.current_position == arenagoal:
			cost = current.cost
			astar_path = []
			while current:
				astar_path.append(arena[current.current_position[0]][current.current_position[1]])
				current = parent[current]
			endram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
			goalDepth = len(astar_path)
			endtime = time.time()
			maxram = endram - startram
			runtime = endtime - starttime
			return astar_path, cost, -1, -1, -1, runtime, maxram
		
		for valid_move in current.expand():
			'''heuristic = abs(valid_move.current_position[0] - arenagoal[0]) + abs(valid_move.current_position[1] - arenagoal[1])
			function = valid_move.cost + heuristic
			valid_move.cost = function'''

			if valid_move not in path and valid_move.current_position not in explored:
				frontier.put((valid_move.cost, valid_move))
				explored.add(valid_move.current_position)
				parent[valid_move] = current
			#elif valid_move in path:

				

	#print(arenastart)
	#print(arenagoal)
	endram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	endtime = time.time()
	runtime = endtime - starttime
	maxram = endram - startram
	return [], -1, -1, -1, -1, runtime, maxram# Replace with return values
	#=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#
	
'''
This function runs Iterative Deepening A* Search on the input arena (which is a list of str)
Returns a ([], int) tuple where the [] represents the solved arena as a list of str and the int represents the cost of the solution
'''
def ida(arena):

	#=================================================#
	#*#*#*# TODO: Write your IDA algorithm here #*#*#*#
	#=================================================#
	startram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	starttime = time.time()

	currstate = MazeState(arena)
	arenastart = currstate.start
	arenagoal = currstate.goal
	#print(arenastart)
	#print(arenagoal)
	endram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	endtime = time.time()
	runtime = endtime - starttime
	maxram = endram - startram
	return [], -1, -1, -1, -1, runtime, maxram# Replace with return values
	#=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#

if __name__ == "__main__":
	if results.bfs:
		print("\nBFS algorithm called")
		bfs_arena, bfs_cost, bfs_nodes_expanded, bfs_max_nodes_stored, bfs_max_search_depth, bfs_time, bfs_ram = bfs(arena)
		print("\n".join(bfs_arena))
		print("BFS:")
		print("Cost: " + str(bfs_cost))
		print("Nodes Expanded: " + str(bfs_nodes_expanded))
		print("Max Nodes Stored: " + str(bfs_max_nodes_stored))
		print("Max Search Depth: " + str(bfs_max_search_depth))
		print("Time: " + str(bfs_time) + "s")
		print("RAM Usage: " + str(bfs_ram) + "kB\n")

	if results.dfs:
		print("\nDFS algorithm called")
		dfs_arena, dfs_cost, dfs_nodes_expanded, dfs_max_nodes_stored, dfs_max_search_depth, dfs_time, dfs_ram = dfs(arena)
		print("\n".join(dfs_arena))
		print("DFS:")
		print("Cost: " + str(dfs_cost))
		print("Nodes Expanded: " + str(dfs_nodes_expanded))
		print("Max Nodes Stored: " + str(dfs_max_nodes_stored))
		print("Max Search Depth: " + str(dfs_max_search_depth))
		print("Time: " + str(dfs_time) + "s")
		print("RAM Usage: " + str(dfs_ram) + "kB\n")

	if results.astar:
		print("\nA* algorithm called")
		astar_arena, astar_cost, astar_nodes_expanded, astar_max_nodes_stored, astar_max_search_depth, astar_time, astar_ram = astar(arena)
		print("\n".join(astar_arena))
		print("A*:")
		print("Cost: " + str(astar_cost))
		print("Nodes Expanded: " + str(astar_nodes_expanded))
		print("Max Nodes Stored: " + str(astar_max_nodes_stored))
		print("Max Search Depth: " + str(astar_max_search_depth))
		print("Time: " + str(astar_time) + "s")
		print("RAM Usage: " + str(astar_ram) + "kB\n")
	
	if results.ida:
		print("\nIterative Deepening A* algorithm called")
		ida_arena, ida_cost, ida_nodes_expanded, ida_max_nodes_stored, ida_max_search_depth, ida_time, ida_ram = ida(arena)
		print("\n".join(ida_arena))
		print("Iterative Deepening A*:")
		print("Cost: " + str(ida_cost))
		print("Nodes Expanded: " + str(ida_nodes_expanded))
		print("Max Nodes Stored: " + str(ida_max_nodes_stored))
		print("Max Search Depth: " + str(ida_max_search_depth))
		print("Time: " + str(ida_time) + "s")
		print("RAM Usage: " + str(ida_ram) + "kB\n")

