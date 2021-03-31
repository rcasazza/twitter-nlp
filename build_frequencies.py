import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np

from utils import process_tweet, build_freqs

nltk.download('stopwords')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

tweets = all_positive_tweets + all_negative_tweets

print("Number of tweets: ", len(tweets))

# first 5000 are positive tweets, the second 5000 are negative tweets
labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

# create a dictionary
# (word, 1.0): integer - map word in positive tweets to count found in all positive tweets
# (word, 0.0): integer - map word in negative tweets to count found in all negative tweets
freqs = build_freqs(tweets, labels)
print(f'type(freqs) = {type(freqs)}')
print(f'len(freqs) = {len(freqs)}')

keys = ['happi', 'merri', 'nice', 'good', 'bad', 'sad', 'mad', 'best', 'pretti',
        '❤', ':)', ':(', '😒', '😬', '😄', '😍', '♛',
        'song', 'idea', 'power', 'play', 'magnific']

data = []

for word in keys:

    # initialize positive and negative counts
    pos = 0
    neg = 0

    # retrieve number of positive counts
    if (word, 1) in freqs:
        pos = freqs[(word, 1)]

    # retrieve number of negative counts
    if (word, 0) in freqs:
        neg = freqs[(word, 0)]

    # append the word counts to the table
    data.append([word, pos, neg])

#print(data)

fig, ax = plt.subplots(figsize = (8, 8))

# Plot in the logarithmic scale to take into account the wide discrepancies between the raw counts
# (e.g. :) has 3568 counts in the positive while only 2 in the negative).

# convert positive raw counts to logarithmic scale. we add 1 to avoid log(0)
x = np.log([x[1] + 1 for x in data])

# do the same for the negative counts
y = np.log([x[2] + 1 for x in data])

# Plot a dot for each pair of words
ax.scatter(x, y)

# assign axis labels
plt.xlabel("Log Positive count")
plt.ylabel("Log Negative count")

# Add the word as the label at the same position as you added the points just before
for i in range(0, len(data)):
    ax.annotate(data[i][0], (x[i], y[i]), fontsize=12)

ax.plot([0, 9], [0, 9], color = 'red') # Plot the red line that divides the 2 areas.
plt.show()










