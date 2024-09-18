# coms-4701
AI assignments

if currpos == "g":
			goalpath.append(currpos)
			break

		goalpath.append(currpos)

		for nextpos in arena[currpos]:
			upPos = currMaze.move_up(currpos) 
			if upPos != None:
				nextpos = upPos
				nodequeue.append(nextpos)
			rightPos = currMaze.move_right(currpos)
			elif rightPos != None:
				nextpos = rightPos
				nodequeue.append(nexpos)
			downPos = currMaze.move_down(currpos)
			elif downPos  != None:
				nextpos = downPos
				nodequeue.append(nextpos)
			leftPos = currMaze.move_left(currpos)
			elif leftPos != None:
				nextpos = leftPos
				nodequeue.append(nextpos)
