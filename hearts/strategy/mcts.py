import random
import math
import hashlib


"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf

The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.

The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  

In particular there are two models of best child that one can use 
"""


class IState(object):

    def next_state(self):
        raise NotImplementedError()

    def terminal(self):
        raise NotImplementedError()

    def reward(self):
        raise NotImplementedError()


class State(IState):
    NUM_TURNS = 10    
    GOAL = 0
    MOVES=[2,-2,3,-3]
    MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
    num_moves=len(MOVES)
    
    def __init__(self, value=0, moves=[], turn=NUM_TURNS):
        self.value = value
        self.turn = turn
        self.moves = moves
    
    def next_state(self):
        next_move = random.choice([x*self.turn for x  in self.MOVES])
        next_state = State(self.value+next_move, self.moves+[next_move], self.turn-1)
        return next
    
    def terminal(self):
        if self.turn == 0:
            return True
        return False
    
    def reward(self):
        r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
        return r
    
    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)
    
    def __eq__(self,other):
        if hash(self) == hash(other):
            return True
        return False
    
    def __repr__(self):
        s="Value: %d; Moves: %s"%(self.value,self.moves)
        return s
    

class Node():

    def __init__(self, state, parent=None):
        self.visits=1
        self.reward=0.0    
        self.state=state
        self.children=[]
        self.parent=parent    
    
    def add_child(self, child_state):
        for child in self.children:
            if child.state == child_state:
                return
        child=Node(child_state, self)
        self.children.append(child)
 
    def move_to_child(self, child_state):
        self.add_child(child_state)
        for child in self.children:
            if child_state == child_state:
                return child
        print('can not find child')
        return None

    def update(self, reward):
        self.reward+=reward
        self.visits+=1
    
    def fully_expanded(self):
        if len(self.children) == self.state.num_moves:
            return True
        return False
    
    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
        return s
        

class MCTS(object):
    
    def __init__(self, budget):
        self._budget = budget
        self.SCALAR = 1 / math.sqrt(2.0) # larger scalar will increase exploitation, smaller will increase exploration

    def UCTSEARCH(self, root):
        for iter in range(int(self._budget)):
            front = self.TREEPOLICY(root)
            reward = self.DEFAULTPOLICY(front.state)
            self.BACKUP(front, reward)
        return self.BESTCHILD(root, 0)

    def TREEPOLICY(self, node):
        #a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
        while node.state.terminal() is False:
            if len(node.children) == 0:
                return self.EXPAND(node)
            elif random.uniform(0,1) < .5: # exploitation
                node=self.BESTCHILD(node, self.SCALAR)
            elif node.fully_expanded() is False: # exploration   
                return self.EXPAND(node)
            else:  
                node=self.BESTCHILD(node, self.SCALAR)
        return node

    def EXPAND(self, node):
        # find out the un-expanded state
        tried_children=[c.state for c in node.children]
        new_state = node.state.next_state()
        while new_state in tried_children:
            new_state = node.state.next_state()
        node.add_child(new_state)
        return node.children[-1]

    #current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
    def BESTCHILD(self, node, scalar):
        bestscore=0.0
        bestchildren=[]
        for c in node.children:
            exploit = c.reward / c.visits
            explore = math.sqrt(2.0*math.log(node.visits) / float(c.visits))    
            score = exploit+scalar*explore
            if score == bestscore:
                bestchildren.append(c)
            if score > bestscore:
                bestchildren = [c]
                bestscore = score
        if len(bestchildren) == 0:
            print("OOPS: no best child found, probably fatal")
        return random.choice(bestchildren)

    def DEFAULTPOLICY(self, state):
        while state.terminal() is False:
            state = state.next_state()
        return state.reward()

    def BACKUP(self, node, reward):
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent
