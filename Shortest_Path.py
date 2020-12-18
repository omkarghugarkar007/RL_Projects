import numpy as np

gamma = 0.9
alpha = 0.9

location_to_state = {
    'A':0,
    'B':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'H':7,
    'I':8,
    'J':9,
    'K':10,
    'L':11,
    'M':12,
    'N':13,
    'O':14,
    'P':15
}

actions = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

R = np.array([[0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
             [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
             [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0],
             [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
             [0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0],
             [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0],
             [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
             [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
             [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0],
             [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1],
             [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0]])


#print(R.shape)
state_to_location = {state: location for location, state in location_to_state.items()}

print("Enter Starting Location:")
starting_location = input()
print("Enter Ending Location:")
ending_location = input()
goal_state = location_to_state[ending_location]
R[goal_state,goal_state] = 100
print(R)

Q = np.zeros((16,16))

for i in range(1000):
    current_state = np.random.randint(0,16)
    playable_action = []
    for j in range(16):
        if R[current_state,j] > 0:
            playable_action.append(j)
    next_state = np.random.choice(playable_action)
    TD = R[current_state,next_state] + gamma*Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
    Q[current_state,next_state] = Q[current_state,next_state] + alpha*TD

print("Q_Values")
print(Q.astype(int))

def route(starting_location, ending_location):

    route = [starting_location]
    next_location = starting_location

    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location

    return route

print("Route:")
print(route(starting_location,ending_location))
