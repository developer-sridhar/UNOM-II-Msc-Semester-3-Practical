import csv

a = []
with open('training_data.csv', 'r') as csvfile:
    for row in csv.reader(csvfile):
        a.append(row)

print("\nThe Given Training Data Set\n", a)
print("\nThe Total Number of Training Instances are: ", len(a))

num_attribute = len(a[0]) - 1
print("\nThe Initial Hypothesis is:")
hypothesis = ['0'] * num_attribute
print(hypothesis)

for i in range(len(a)):
    if a[i][-1].lower() == 'yes':
        for j in range(num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:
                hypothesis[j] = a[i][j]
            else:
                hypothesis[j] = '?'
        print("\nThe Hypothesis for the training instance {} is:\n".format(i + 1), hypothesis)

print("\nThe Maximally Specific Hypothesis for the training data is:")
print(hypothesis)
