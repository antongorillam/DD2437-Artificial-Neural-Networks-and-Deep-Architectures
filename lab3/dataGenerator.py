import numpy as np

def getW(patterns,selfConnect = False, scaling = False):

    W = np.zeros((len(patterns[0]),len(patterns[0])))
    for pattern in patterns: 
        wi = np.outer(pattern, pattern)
        if not selfConnect:
            np.fill_diagonal(wi, 0)
        W += wi
    if scaling:
        W = (1/(len(patterns[0]))) * W
    return W

def getW36(patterns,p,selfConnect = False, scaling = False):
    W = np.zeros((len(patterns[0]),len(patterns[0])))
    for pattern in patterns: 
        #print('pattern: ',pattern)
        #print('p: ',p)
        #print('pattern - p', pattern-p)
        wi = np.outer(pattern-p, pattern-p)
        if not selfConnect:
            np.fill_diagonal(wi, 0)
        W += wi
    if scaling:
        W = (1/(len(patterns[0]))) * W
    return W

def getPatterns(directory, neurons):
    file = open(directory, 'r')
    all_data = []
    for line in file.readlines():
        data = line.rstrip().split(',')
        data = [int(i) for i in data]
        all_data = np.concatenate([all_data,data])
    patterns = all_data.reshape(int(all_data.size/neurons), neurons)
    return patterns


if __name__ == "__main__":
    patterns = getPatterns(directory='pat31.dat', neurons=8)
    print(patterns)
    W = getW(patterns)
    print(W)
    print(f'x1: {np.sign(W @ patterns[0])}')