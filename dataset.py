"""
line format:
1241160900 109513 0 |user 2:0.000012 3:0.000000 4:0.000006 5:0.000023 6:0.999958 1:1.000000 |109498 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 |109509 2:0.306008 3:0.000450 4:0.077048 5:0.230439 6:0.386055 1:1.000000 [[...more article features omitted...]] |109453 2:0.421669 3:0.000011 4:0.010902 5:0.309585 6:0.257833 1:1.000000

Some log files contain rows with erroneous data

After the first 10 columns are the articles and their features.
Each article has 7 columns (articleid + 6 features)
Therefore number_of_columns-10 % 7 = 0
"""
import numpy as np
import fileinput


def get_yahoo_events(filenames):
    global articles, features, events, n_arms, n_events
    articles = []
    features = []
    events = []
    n_arms = 0
    n_events = 0

    skipped = 0

    with fileinput.input(files=filenames) as f:
        for line in f:
            for rem in ["1:", "2:", "3:", "4:", "5:", "6:", "|"]:
                line = line.replace(rem, "")

            cols = line.split()
            if (len(line.split()) - 10) % 7 != 0:
                skipped += 1
            else:
                pool_idx = []
                pool_ids = []
                for i in range(10, len(cols) - 6, 7):
                    if cols[i] not in articles:
                        articles.append(cols[i])
                        features.append([float(x) for x in cols[i + 1 : i + 7]])
                    pool_idx.append(articles.index(cols[i]))
                    pool_ids.append(cols[i])

                events.append(
                    [
                        pool_ids.index(cols[1]),
                        int(cols[2]),
                        [float(x) for x in cols[4:10]],
                        pool_idx,
                    ]
                )
    features = np.array(features)
    n_arms = len(articles)
    n_events = len(events)
    print(n_events, "events with", n_arms, "articles")
    if skipped != 0:
        print("Skipped events:", skipped)
    """
        articles : [article_ids]
        features : [[article_1_features] .. [article_n_features]]
        events : [
             0 displayed_article_index (relative to the pool),
             1 user_click,
             2 [user_features],
             3 [pool_indexes]
    """


def max_articles(n_articles):
    global articles, features, events, n_arms, n_events
    n_arms = n_articles
    articles = articles[:n_articles]
    features = features[:n_articles]
    events_reduced = []
    for event in events:
        displayed_idx = event[0]  # index relative to the pool
        if displayed_idx < n_articles:
            event[3] = np.arange(0, n_arms)  # pool = all available articles
            events_reduced.append(event)

    events = events_reduced
    n_events = len(events)
    print("Number of events:", n_events)


# articles, features, events = process_events("ydata-fp-td-clicks-v1_0.20090509")
