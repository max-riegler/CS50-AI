import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Define dummy probability
    jprob = 1

    # Define a helper dictionary that collects info on parents (True/False), numer of genes (0,1,2), and the trait (True/False)
    helper = { x : { "parents" : True , "genes" : 0 , "trait" : False} for x in people}
    
    for person in people:
        if not people[person]["mother"]:
            helper[person]["parents"] = False
        if person in one_gene:
            helper[person]["genes"] = 1
        if person in two_genes:
            helper[person]["genes"] = 2
        if person in have_trait:
            helper[person]["trait"] = True            


    for person in helper:
        
        # Define dummy variable to keep the code readable
        parents = helper[person]["parents"]
        genes = helper[person]["genes"]
        trait = helper[person]["trait"]

        # No info on parents
        if not parents:
            jprob *= PROBS["gene"][genes] * PROBS["trait"][genes][trait]

        # With info on parents
        else:

            # Define dummy variables
            mother = people[person]["mother"]
            father = people[person]["father"]
            inherit_prob = {}
            
            # Determine the individual probabilities a gene that a parent with a specific number of genes passes theirs on
            for parent in [mother, father]:
                if helper[parent]["genes"] == 0:
                    individual_prob = PROBS["mutation"]
                elif helper[parent]["genes"] == 1:
                    individual_prob = 0.5
                else:
                    individual_prob = 1 - PROBS["mutation"]
                inherit_prob[parent] = individual_prob                    

            if genes == 0:
                jprob *= (1 - inherit_prob[mother]) * (1 - inherit_prob[father])
            elif genes == 1:
                jprob *= (1 - inherit_prob[mother]) * inherit_prob[father] + inherit_prob[mother] * (1 - inherit_prob[father])
            else:
                jprob *= inherit_prob[mother] * inherit_prob[father]

            jprob *= PROBS["trait"][genes][trait]
    
    return jprob

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    
    for person in probabilities:
        genes = 1 if person in one_gene else 2 if person in two_genes else 0
        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Define normalization factors
        norm_genes = sum(probabilities[person]["gene"].values())
        norm_traits = sum(probabilities[person]["trait"].values())

        for genes in probabilities[person]["gene"]:
            probabilities[person]["gene"][genes] /= norm_genes
        for traits in probabilities[person]["trait"]:
            probabilities[person]["trait"][traits] /= norm_traits


if __name__ == "__main__":
    main()
