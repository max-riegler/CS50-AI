import sys

import copy

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        # Make a deepcopy of self.domains:
        dummy_domain = copy.deepcopy(self.domains)

        # Iterate through all variables in the domain:
        for variable in dummy_domain:
            var_length = variable.length

            # Iterate through the words in domain and remove words whose length are inconsistent with the constraints:
            for word in dummy_domain[variable]:
                if len(word) != var_length:
                    self.domains[variable].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        
        revised = False

        overlap = self.crossword.overlaps[x,y]

        for X in self.domains[x]:
            count = 0
            for Y in self.domains[y]:

                # If variables do not have overlap: check if the domain of y has other words that the word that is currently checked in the domain of x:
                if overlap == None and X != Y:
                    count += 1

                # If the variables have overlap: Check if there is an element in the domain of y that is consistent with the constraint:
                elif overlap != None and X[overlap[0]] == Y[overlap[1]]:
                    count += 1

            if count == 0:
                self.domains[x].remove(X)
                revised = True
        
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if not arcs:
            queue = []
            for x in self.domains:
                for y in self.crossword.neighbors(x):
                    if self.crossword.overlaps[x, y] is not None:
                        queue.append((x, y))
        
        while arcs:
            x, y = arcs.pop()

            # Revise the arc:
            if self.revise(x, y):

                # If the domain of x is empty, then there is no solution:
                if not self.domains[x]:
                    return False
                
                # If revised, add all x neighbors to arcs:
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for variable in self.domains:
            if variable not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # Check whether or not all values are distinct by making use of set():
        dummy = [x for x in assignment.values()]
        if len(dummy) != len(set(dummy)):
            return False

        # Check if every value has the correct length:
        for variable in assignment:
            if variable.length != len(assignment[variable]):
                return False

        # Check if there are any conflicts between neighboring variables:
        for variable in assignment:
            for neighbor in self.crossword.neighbors(variable):
                if neighbor in assignment:
                    x, y = self.crossword.overlaps[variable, neighbor]
                    if assignment[variable][x] != assignment[neighbor][y]:
                        return False
                    
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        dummy_values = {value: 0 for value in self.domains[var]}

        # Iterate through all values of var:
        for value in self.domains[var]:

            # Iterate through neighboring variables and values:
            for var2 in self.crossword.neighbors(var):
                for value2 in self.domains[var2]:

                    # If a value excludes another value, increment the dummy_values count by 1:
                    if not self.overlap_satisfied(var, var2, value, value2):
                        dummy_values[value] += 1

        # Return a sorted list of values:
        return sorted([x for x in dummy_values], key = lambda x: dummy_values[x])

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # Determine the set of unassigned variables:
        unassigned = set(self.domains.keys()) - set(assignment.keys())

        # Create a sorted list of variables:
        result = [variable for variable in unassigned]
        result.sort(key = lambda x: (len(self.domains[x]), -len(self.crossword.neighbors(x))))

        return result[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # If the assignment is already complete:
        if len(assignment) == len(self.domains):
            return assignment

        # Select one of the unassigned variables:
        variable = self.select_unassigned_variable(assignment)

        # Iterate through all words in the domain of that variable:
        for value in self.domains[variable]:

            # Make a dummy copy with an updated variable value:
            dummy_copy = copy.deepcopy(assignment)
            dummy_copy[variable] = value

            # Check for consistency:
            if self.consistent(dummy_copy):
                result = self.backtrack(dummy_copy)
                if result is not None:
                    return result
        return None

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
