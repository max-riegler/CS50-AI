import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Create base transition
    transition = { x : (1 - damping_factor) / (len(corpus.keys())) for x in corpus.keys()}

    if len(corpus[page]) < 1:
        # If page does not have any links return a probability distribution that chooses randomly among all pages with equal probability
        return { x : 1 / (len(corpus.keys())) for x in corpus.keys()}
    
    else:
        for x in corpus[page]:
            transition[x] += damping_factor / (len(corpus[page]))
        return transition

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Create dictionary with zero pagerank for every entry
    pagerank = {x : 0 for x in corpus.keys()}

    # Starting point
    initial_page = None

    # Iterating n times
    for _ in range(n):
        if initial_page:
            # Previous sample is available, choose the next using the transition model
            model = transition_model(corpus, initial_page, damping_factor)
            transition_keys = list(model.keys())
            transition_probabilities =  [model[i] for i in model]
            initial_page = random.choices(transition_keys, transition_probabilities, k=1)[0]
        else:
            # No previous sample, choose randomly
            initial_page = random.choice(list(corpus.keys()))

        # Count each page
        pagerank[initial_page] += 1

    # Normalize count to 1
    for x in pagerank:
        pagerank[x] /= n

    # Return result
    return pagerank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Create dictionary with 1/N pagerank for every entry
    pagerank = {x : 1 / len(corpus) for x in corpus.keys()}
    NewPagerank = {}

    while True:
        for p in corpus.keys():
            PRNumLinksi = []
            for x in corpus.keys():
                if p in corpus[x]:
                    PRNumLinksi.append(pagerank[x] / len(corpus[x]))
                if len(corpus[x]) == 0:
                    PRNumLinksi.append(pagerank[x] / len(corpus))
            NewPagerank[p] = ((1 - damping_factor) / len(corpus)) + damping_factor * sum(PRNumLinksi)

        difference = max([abs(NewPagerank[x] - pagerank[x]) for x in pagerank])
        if difference < 0.001:
            break
        else:
            pagerank = NewPagerank.copy()

    return pagerank            

if __name__ == "__main__":
    main()