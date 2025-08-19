from functools import reduce
from collections import defaultdict

def mapper(text):
    # Split the text into individual words and return a list of (word, 1) tuples
    return [(word, 1) for word in text.split()]

def reducer(word_counts, pair):
    # Increment the count of the word in the word_counts dictionary
    word, count = pair
    word_counts[word] += count
    return word_counts

# Create a list of input texts (sentences or documents) to process.
input_texts = [
    "Hello world",
    "MapReduce is powerful",
    "Hello world and MapReduce",
    "Hello how are you"
]

# Step 1: Map - apply the mapper function to each input text
mapped_data = map(mapper, input_texts)

# Step 2: Flatten the mapped data into a single list of (word, 1) tuples
flattened_data = [pair for sublist in mapped_data for pair in sublist]

# Step 3: Reduce - apply the reducer function to the list of tuples to get the word counts
word_counts = reduce(reducer, flattened_data, defaultdict(int))

# Step 4: Display the word counts
print(dict(word_counts))
