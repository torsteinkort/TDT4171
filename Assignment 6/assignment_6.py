import numpy as np
from pathlib import Path
from typing import Tuple



class Node:
    """ Node class used to build the decision tree"""
    def __init__(self):
        self.children = {}
        self.parent = None
        self.attribute = None
        self.value = None

    def classify(self, example):
        if self.value is not None:
            return self.value
        return self.children[example[self.attribute]].classify(example)



def plurality_value(examples: np.ndarray) -> int:
    """Implements the PLURALITY-VALUE (Figure 19.5)"""
    labels = np.array(examples)[:, -1]
    value, count = 0, 0
    for label in np.unique(labels):
        label_count = np.count_nonzero(labels == label)
        if label_count > count:
            value = label
            count = label_count

    return value


def importance(attributes: np.ndarray, examples: np.ndarray, measure: str) -> int:
    """
    This function should compute the importance of each attribute and choose the one with highest importance,
    A ← argmax a ∈ attributes IMPORTANCE (a, examples) (Figure 19.5)

    Parameters:
        attributes (np.ndarray): The set of attributes from which the attribute with highest importance is to be chosen
        examples (np.ndarray): The set of examples to calculate attribute importance from
        measure (str): Measure is either "random" for calculating random importance, or "information_gain" for
                        caulculating importance from information gain (see Section 19.3.3. on page 679 in the book)

    Returns:
        (int): The index of the attribute chosen as the test

    """

    # TODO implement the importance function for both measure = "random" and measure = "information_gain"

    if not measure == 'information_gain' and not measure == 'random':
        raise Exception()
    
    # measure = "random"
    if measure == 'random':
        return np.random.randint(len(attributes))
    
    # measure = "information_gain"

    D = 2
    
    def count_values(examples, value):
        count = 0
        for i in range(len(examples)):
            for j in range(len(examples[i])):
                if examples[i][j] == value:
                    count += 1
        return count

    def B(q):
        return - (q * np.log2(q) + (1 - q) * np.log2(1 - q))
    
    def subset_examples(examples, attribute, value):
        return [example for example in examples if example[attribute] == value]
    
    def Remainder(p, n, examples, attribute):
        remainder = 0
        for value in [POSITIVE_VALUE, NEGATIVE_VALUE]:
            p_k = count_values(subset_examples(examples, attribute, value), POSITIVE_VALUE)
            n_k = count_values(subset_examples(examples, attribute, value), NEGATIVE_VALUE)
            if not p_k == 0 and not n_k == 0:
                remainder += ((p_k + n_k)/(p + n)) * B(p_k/(p_k + n_k))
        return remainder

    # calculate each gain
    p = count_values(examples, POSITIVE_VALUE)
    n = count_values(examples, NEGATIVE_VALUE)

    gain_array = []
    for attribute in attributes:
        gain = B(p/(p + n)) - Remainder(p, n, examples, attribute)
        gain_array.append(gain)
    
    # return index of highest gain
    return gain_array.index(max(gain_array))

def learn_decision_tree(examples: np.ndarray, attributes: np.ndarray, parent_examples: np.ndarray,
                        parent: Node, branch_value: int, measure: str):
    """
    This is the decision tree learning algorithm. The pseudocode for the algorithm can be
    found in Figure 19.5 on Page 678 in the book.

    Parameters:
        examples (np.ndarray): The set data examples to consider at the current node
        attributes (np.ndarray): The set of attributes that can be chosen as the test at the current node
        parent_examples (np.ndarray): The set of examples that were used in constructing the current node’s parent.
                                        If at the root of the tree, parent_examples = None
        parent (Node): The parent node of the current node. If at the root of the tree, parent = None
        branch_value (int): The attribute value corresponding to reaching the current node from its parent.
                        If at the root of the tree, branch_value = None
        measure (str): The measure to use for the Importance-function. measure is either "random" or "information_gain"

    Returns:
        (Node): The subtree with the current node as its root
    """

    # Creates a node and links the node to its parent if the parent exists
    node = Node()
    if parent is not None:
        parent.children[branch_value] = node
        node.parent = parent

    # TODO implement the steps of the pseudocode in Figure 19.5 on page 678

    # Check if examples is empty
    if len(examples) == 0:
        node.value = plurality_value(parent_examples)
        return node

    # Check if all examples have the same classification
    all_same = all(example[-1] == examples[0][-1] for example in examples)
    if all_same:
        node.value = examples[0][-1]
        return node
    
    # Check if attributes is empty
    if len(attributes) == 0:
        node.value = plurality_value(examples)
        return node


    A = importance(attributes, examples, measure)
    node.attribute = attributes[A]
    for value in [POSITIVE_VALUE, NEGATIVE_VALUE]:
        exs = [example for example in examples if example[A] == value]
        subtree = learn_decision_tree(exs, np.delete(attributes, A), examples, node, value, measure)
        node.children[value] = subtree
    return node



def accuracy(tree: Node, examples: np.ndarray) -> float:
    """ Calculates accuracy of tree on examples """
    correct = 0
    for example in examples:
        pred = tree.classify(example[:-1])
        correct += pred == example[-1]
    return correct / examples.shape[0]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Load the data for the assignment,
    Assumes that the data files is in the same folder as the script"""
    with (Path.cwd() / "train.csv").open("r") as f:
        train = np.genfromtxt(f, delimiter=",", dtype=int)
    with (Path.cwd() / "test.csv").open("r") as f:
        test = np.genfromtxt(f, delimiter=",", dtype=int)
    return train, test

# For simplification, global values
POSITIVE_VALUE = 1
NEGATIVE_VALUE = 2

if __name__ == '__main__':

    train, test = load_data()

    # information_gain or random
    measure = "random"

    tree = learn_decision_tree(examples=train,
                    attributes=np.arange(0, train.shape[1] - 1, 1, dtype=int),
                    parent_examples=None,
                    parent=None,
                    branch_value=None,
                    measure=measure)

    print(f"Training Accuracy {accuracy(tree, train)}")
    print(f"Test Accuracy {accuracy(tree, test)}")