"""
Each futoshiki board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8

Empty values in the board are represented by 0

An * after the letter indicates the inequality between the row represented
by the letter and the next row.
e.g. my_board['A*1'] = '<' 
means the value at A1 must be less than the value
at B1

Similarly, an * after the number indicates the inequality between the
column represented by the number and the next column.
e.g. my_board['A1*'] = '>' 
means the value at A1 is greater than the value
at A2

Empty inequalities in the board are represented as '-'

"""
import sys
import numpy as np
import copy
import time
#======================================================================#
#*#*#*# Optional: Import any allowed libraries you may need here #*#*#*#
#======================================================================#

#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

ROW = "ABCDEFGHI"
COL = "123456789" #a cell is represented as row col -> A1

class Board:
    '''
    Class to represent a board, including its configuration, dimensions, and domains
    '''
    
    def get_board_dim(self, str_len):
        '''
        Returns the side length of the board given a particular input string length
        '''
        d = 4 + 12 * str_len
        n = (2+np.sqrt(4+12*str_len))/6
        if(int(n) != n):
            raise Exception("Invalid configuration string length")
        
        return int(n)
        
    def get_config_str(self):
        '''
        Returns the configuration string
        '''
        return self.config_str
        
    def get_config(self):
        '''
        Returns the configuration dictionary
        '''
        return self.config
        
    def get_variables(self):
        '''
        Returns a list containing the names of all variables in the futoshiki board
        '''
        variables = []
        for i in range(0, self.n):
            for j in range(0, self.n):
                variables.append(ROW[i] + COL[j])
        return variables
    
    def convert_string_to_dict(self, config_string):
        '''
        Parses an input configuration string, retuns a dictionary to represent the board configuration
        as described above
        '''
        config_dict = {}
        
        for i in range(0, self.n): #n is the length for the side of the board
            for j in range(0, self.n):
                cur = config_string[0]
                config_string = config_string[1:]
                
                config_dict[ROW[i] + COL[j]] = int(cur)
                
                if(j != self.n - 1):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + COL[j] + '*'] = cur
                    
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_string[0]
                    config_string = config_string[1:]
                    config_dict[ROW[i] + '*' + COL[j]] = cur
                    
        return config_dict
        
    def print_board(self):
        '''
        Prints the current board to stdout
        '''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    print('_', end=' ')
                else:
                    print(str(cur), end=' ')
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        print(' ', end=' ')
                    else:
                        print(cur, end=' ')
            print('')
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        print(' ', end='   ')
                    else:
                        print(cur, end='   ')
            print('')
    
    def __init__(self, config_string):
        '''
        Initialising the board
        '''
        self.config_str = config_string
        self.n = self.get_board_dim(len(config_string))
        if(self.n > 9):
            raise Exception("Board too big")
            
        self.config = self.convert_string_to_dict(config_string)
        self.domains = self.reset_domains()
        
        self.forward_checking(self.get_variables())
        
        
    def __str__(self):
        '''
        Returns a string displaying the board in a visual format. Same format as print_board()
        '''
        output = ''
        config_dict = self.config
        for i in range(0, self.n):
            for j in range(0, self.n):
                cur = config_dict[ROW[i] + COL[j]]
                if(cur == 0):
                    output += '_ '
                else:
                    output += str(cur)+ ' '
                
                if(j != self.n - 1):
                    cur = config_dict[ROW[i] + COL[j] + '*']
                    if(cur == '-'):
                        output += '  '
                    else:
                        output += cur + ' '
            output += '\n'
            if(i != self.n - 1):
                for j in range(0, self.n):
                    cur = config_dict[ROW[i] + '*' + COL[j]]
                    if(cur == '-'):
                        output += '    '
                    else:
                        output += cur + '   '
            output += '\n'
        return output
        
    def reset_domains(self):
        '''
        Resets the domains of the board assuming no enforcement of constraints
        '''
        domains = {}
        variables = self.get_variables()
        for var in variables:
            if(self.config[var] == 0):
                domains[var] = [i for i in range(1,self.n+1)]
            else:
                domains[var] = [self.config[var]]
                
        self.domains = domains
                
        return domains

    def forward_checking(self, reassigned_variables):
        #Runs the forward checking algorithm to restrict the domains of all variables based on the values
        #of reassigned variables
        #print(reassigned_variables)
        #======================================================================#
		#*#*#*# TODO: Write your implementation of forward checking here #*#*#*#
		#======================================================================#
        for variable in reassigned_variables:
            current = self.config[variable]
            if current == 0:
                continue
            row = variable[0]
            column = variable[1]
            #implementing row constraints 
            for coln in COL[:self.n]:#checking for all in the size of the board
                #print("coln")
                #print(coln)
                next_pos = row + coln
                if variable != next_pos and current in self.domains[next_pos]:
                    self.domains[next_pos].remove(current)
        
            #implementing column constraints
            for rows in ROW[:self.n]:
                # print("row")
                #print(row)
                next_pos = rows + column
                if variable != next_pos and current in self.domains[next_pos]:
                    self.domains[next_pos].remove(current)
        
            #implementing checks for the inequality symbols.
        
            #Horizontal inequalities!
            if column != COL[self.n -1]: #out of bounds condition
                constrvalue = variable + '*'
                symbol = self.config.get(constrvalue, '-') #if the constraint value is 
                #present if not use the - as the default inequality symbol
                #checking for the symbols.
                if symbol != '-':
                    #find the next column val from the reassigned vars
                    next_pos = row + COL[COL.index(column) + 1] 
                    if symbol == '>':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var < current]
                    elif symbol == '<':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var > current]

            if column != COL[0]:
                prevcol = COL[COL.index(column) - 1]
                constrvalue = row + prevcol + '*'
                symbol = self.config.get(constrvalue, '-')
                if symbol != '-':
                    next_pos = row + prevcol    
                    if symbol == '>':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var > current]
                    elif symbol == '<':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var < current]

            #vertical inequalities
            if row != ROW[self.n -1]: #out of bounds condition
                constrvalue = row + '*' + column
                symbol = self.config.get(constrvalue, '-') #if the constraint value is 
                #present if not use the - as the default inequality symbol
                #checking for the symbols.
                if symbol != '-':
                    #find the next column val from the reassigned vars
                    next_pos =  ROW[ROW.index(row) + 1] + column 
                    if symbol == '>':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var < current]
                    elif symbol == '<':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var > current]


            if row != ROW[0]:
                constrvalue =  ROW[ROW.index(row) - 1] + '*' + column
                symbol = self.config.get(constrvalue, '-')
                if symbol != '-':
                    next_pos = ROW[ROW.index(row) - 1] + column   
                    if symbol == '>':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var > current]
                    elif symbol == '<':
                        self.domains[next_pos] = [var for var in self.domains[next_pos] if var < current]

        #=================================#
		#*#*#*# Your code ends here #*#*#*#
		#=================================#
        
    #=================================================================================#
	#*#*#*# Optional: Write any other functions you may need in the Board Class #*#*#*#
	#=================================================================================#
        
    #=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#

#================================================================================#
#*#*#*# Optional: You may write helper functions in this space if required #*#*#*#
#================================================================================#        

#=================================#
#*#*#*# Your code ends here #*#*#*#
#=================================#

def backtracking(board):
    '''
    Performs the backtracking algorithm to solve the board
    Returns only a solved board
    '''
    #print('backtracking')
    #==========================================================#
	#*#*#*# TODO: Write your backtracking algorithm here #*#*#*#
	#==========================================================#
    def unassigned_vars():
        variables = [] #getting all the unsigned variables.
        for val in board.get_variables():
            if board.config[val] == 0: 
                variables.append(val) 
        mindomainlen = float('inf') # getting the smallest domain of the variables! mrv heuristic approach

        for val in variables:
            currlen = len(board.domains[val])
            if currlen < mindomainlen:
                mindomainlen = currlen

        for val in variables:
            if len(board.domains[val]) == mindomainlen:
                return val
        

    def consistencycheck(variable, values):
        #row and column uniqueness
        row = variable[0]
        column = variable[1]
                
        #print(row)
        #print(column)
        
        for coln in COL[:board.n]:
            next_pos = row + coln
            if next_pos != variable and board.config[next_pos] == values:
                return False
        
        for rows in ROW[:board.n]:
            next_pos = rows + column
            if next_pos != variable and board.config[next_pos] == values:
                return False

        #checking an handling the inequalities
        #horizontal
        if column != COL[board.n -1]: #out of bounds condition
            constrvalue = variable + '*'
            symbol = board.config.get(constrvalue, '-') #if the constraint value is 
            #present if not use the - as the default inequality symbol
            #checking for the symbols.
            if symbol != '-':
                #find the next column val from the reassigned vars
                next_pos = row + COL[COL.index(column) + 1] 
                nextval = board.config[next_pos]
                if nextval != 0:
                    if symbol == '>' and values <= nextval:
                            return False
                    elif symbol == '<' and values >= nextval:
                            return False

        if column != COL[0]: #out of bounds condition
            prev_col = COL[COL.index(column) - 1]
            constrvalue = row + prev_col + '*'
            symbol = board.config.get(constrvalue, '-') #if the constraint value is 
            #present if not use the - as the default inequality symbol
            #checking for the symbols.
            if symbol != '-':
                #find the next column val from the reassigned vars
                prev_pos = row + prev_col 
                nextval = board.config[prev_pos]
                if nextval != 0:
                    if symbol == '>' and values >= nextval:
                        print(f"Consistency check failed: {variable} ({values}) < {next_pos} ({nextval})")
                        return False
                    elif symbol == '<' and values <= nextval:
                        print(f"Consistency check failed: {variable} ({values}) < {next_pos} ({nextval})")
                        return False

        #vertical
        #vertical inequalities
        if row != ROW[board.n -1]: #out of bounds condition
            constrvalue = row + '*' + column
            symbol = board.config.get(constrvalue, '-') #if the constraint value is 
            #present if not use the - as the default inequality symbol
            #checking for the symbols.
            if symbol != '-':
                #find the next column val from the reassigned vars
                next_pos = ROW[ROW.index(row) + 1] + column
                nextval = board.config[next_pos]
                if nextval != 0:
                    if symbol == '>' and values <= nextval:
                        print(f"Consistency check failed: {variable} ({values}) < {next_pos} ({nextval})")
                        return False
                    elif symbol == '<' and values >= nextval:
                        print(f"Consistency check failed: {variable} ({values}) < {next_pos} ({nextval})")
                        return False

        if row != ROW[0]: #out of bounds condition
            constrvalue = ROW[ROW.index(row) - 1]  + '*' + column
            symbol = board.config.get(constrvalue, '-') #if the constraint value is 
            #present if not use the - as the default inequality symbol
            #checking for the symbols.
            if symbol != '-':
                #find the next column val from the reassigned vars
                next_pos = ROW[ROW.index(row) - 1] + column
                nextval = board.config[next_pos]
                if nextval != 0:
                    if symbol == '>' and values >= nextval:
                            #print(f"Consistency check failed: {variable} ({values}) < {next_pos} ({nextval})")
                            return False
                    elif symbol == '<' and values <= nextval:
                            #print(f"Consistency check failed: {variable} ({values}) < {next_pos} ({nextval})")
                            return False
        
        return True

    def backtrack():
        #base case -> check if the board is completely filled
        if all(board.config[val] != 0 for val in board.get_variables()):
            return True

        curr_var = unassigned_vars()
       
        vals = board.domains[curr_var]

        for val in vals:
            if consistencycheck(curr_var, val):
                board.config[curr_var] = val
                domain_tracked = copy.deepcopy(board.domains)
                board.domains[curr_var] = [val]
                board.forward_checking([curr_var])

                if all(len(board.domains[var]) > 0 for var in board.get_variables()):
                    solveboard = backtrack()
                    if solveboard:
                        return True
                #unassigned vals and restore the domain before change
                board.config[curr_var] = 0
                board.domains = domain_tracked
        return False

    if backtrack():
        return board
    else:
        return None

    #return None # Replace with return values
    #=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#
    
def solve_board(board):
    '''
    Runs the backtrack helper and times its performance.
    Returns the solved board and the runtime
    '''
    #================================================================#
	#*#*#*# TODO: Call your backtracking algorithm and time it #*#*#*#
	#================================================================#
    start = time.time()
   # print(start)  
    soln  = backtracking(board)
    endtime = time.time()
    runtime = endtime - start
    #print(runtime)
    return soln, runtime # Replace with return values
    #=================================#
	#*#*#*# Your code ends here #*#*#*#
	#=================================#

def print_stats(runtimes):
    '''
    Prints a statistical summary of the runtimes of all the boards
    '''
    min = 100000000000
    max = 0
    sum = 0
    n = len(runtimes)

    for runtime in runtimes:
        sum += runtime
        if(runtime < min):
            min = runtime
        if(runtime > max):
            max = runtime

    mean = sum/n

    sum_diff_squared = 0

    for runtime in runtimes:
        sum_diff_squared += (runtime-mean)*(runtime-mean)

    std_dev = np.sqrt(sum_diff_squared/n)

    print("\nRuntime Statistics:")
    print("Number of Boards = {:d}".format(n))
    print("Min Runtime = {:.8f}".format(min))
    print("Max Runtime = {:.8f}".format(max))
    print("Mean Runtime = {:.8f}".format(mean))
    print("Standard Deviation of Runtime = {:.8f}".format(std_dev))
    print("Total Runtime = {:.8f}".format(sum))


if __name__ == '__main__':
    if len(sys.argv) > 1:

        # Running futoshiki solver with one board $python3 futoshiki.py <input_string>.
        print("\nInput String:")
        print(sys.argv[1])
        
        print("\nFormatted Input Board:")
        board = Board(sys.argv[1])
        board.print_board()
        
        solved_board, runtime = solve_board(board)
        
        print("\nSolved String:")
        print(solved_board.get_config_str())
        
        print("\nFormatted Solved Board:")
        solved_board.print_board()
        
        print_stats([runtime])

        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(solved_board.get_config_str())
        outfile.write('\n')
        outfile.close()

    else:
        # Running futoshiki solver for boards in futoshiki_start.txt $python3 futoshiki.py

        #  Read boards from source.
        src_filename = 'futoshiki_start.txt'
        try:
            srcfile = open(src_filename, "r")
            futoshiki_list = srcfile.read()
            srcfile.close()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()

        # Setup output file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        
        runtimes = []

        # Solve each board using backtracking
        for line in futoshiki_list.split("\n"):
            
            print("\nInput String:")
            print(line)
            
            print("\nFormatted Input Board:")
            board = Board(line)
            board.print_board()
            
            solved_board, runtime = solve_board(board)
            runtimes.append(runtime)
            
            print("\nSolved String:")
            print(solved_board.get_config_str())
            
            print("\nFormatted Solved Board:")
            solved_board.print_board()

            # Write board to file
            outfile.write(solved_board.get_config_str())
            outfile.write('\n')

        # Timing Runs
        print_stats(runtimes)
        
        outfile.close()
        print("\nFinished all boards in file.\n")