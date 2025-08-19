import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('training_data.csv')

# Extract features and target
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i].lower() == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        elif target[i].lower() == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]
    return specific_h, general_h

# Run learning algorithm
s_final, g_final = learn(concepts, target)

# Final Output
print("Final Specific_h:")
print(s_final)

print("Final General_h:")
if g_final:
    for h in g_final:
        print(h)
else:
    print("No consistent general hypotheses found.")
