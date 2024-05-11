import functools

import numpy as np
from ontolearn.owlapy.model import OWLClassExpression, OWLObjectComplementOf, OWLObjectUnionOf, \
    OWLObjectIntersectionOf, OWLObjectSomeValuesFrom, OWLObjectAllValuesFrom, OWLObjectMaxCardinality, \
    OWLObjectMinCardinality, OWLClass, OWLObjectProperty, OWLDataSomeValuesFrom, OWLObjectOneOf
from torch_geometric.data import HeteroData


class Evaluator:
    """ An evaluator which is able to evaluate the accuracy of a given logical formula based on a given dataset."""

    def __init__(self, data: HeteroData):
        """"
        Initializes the evaluator based on the given dataset. After the initialization the object should be able to
        evaluate logical formulas based on the dataset.

        Args:
            data: The dataset which should be used for evaluation.
        """""
        self._data = data
        self._nodeset = self._get_nodeset()

        self.owl_mapping = {
            OWLObjectComplementOf: self._eval_complement,
            OWLObjectUnionOf: self._eval_union,
            OWLObjectIntersectionOf: self._eval_intersection,
            OWLObjectSomeValuesFrom: self._eval_existential,
            OWLObjectAllValuesFrom: self._eval_universal,
            OWLObjectMaxCardinality: self._eval_max_cardinality,
            OWLObjectMinCardinality: self._eval_min_cardinality,
            OWLClass: self._eval_class,
            OWLDataSomeValuesFrom: self._eval_property_value,
            OWLObjectOneOf: self._eval_object_one_of
        }

    @property
    def data(self) -> HeteroData:
        """
        The dataset which should be used for evaluation.

        Returns:
            The dataset which should be used for evaluation.
        """
        return self._data

    @data.setter
    def data(self, val: HeteroData) -> None:
        """
        Sets the dataset which should be used for evaluation to the given value.

        Args:
            val: The dataset which should be used for evaluation.
        """
        self._data = val

    def explanation_accuracy(self, ground_truth: set[tuple[int, str]],
                             logical_formula: OWLClassExpression) -> tuple[float, float, float]:
        """
        Calculates the explanation accuracy of the given logical formula based on the given ground truth.

        Args:
            ground_truth: The ground truth which should be used for evaluation.
            logical_formula: The logical formula which should be evaluated.

        Returns:
            A triple containing the precision, the recall and the accuracy of the given logical formula based on the given
            ground truth.
        """
        tp, fp, tn, fn = self._get_positive_negatives(ground_truth, logical_formula)
        if tp + fp == 0:
            return 0, tp / (tp + fn), (tp + tn) / (tp + fp + tn + fn)

        return tp / (tp + fp), tp / (tp + fn), (tp + tn) / (tp + fp + tn + fn)

    def f1_score(self, ground_truth: set[tuple[int, str]], logical_formula: OWLClassExpression) -> float:
        """
        Calculates the F1 score of the given logical formula based on the given ground truth.
        Args:
            ground_truth: The ground truth which should be used for evaluation.
            logical_formula: The logical formula which should be evaluated.

        Returns:
            The F1 score of the given logical formula based on the given ground truth.
        """
        tp, fp, _, fn = self._get_positive_negatives(ground_truth, logical_formula)

        return (2 * tp) / (2 * tp + fp + fn)

    def _get_positive_negatives(self, ground_truth: set[tuple[int, str]], logical_formula: OWLClassExpression) \
            -> tuple[float, float, float, float]:
        """
        Calculates the sizes of the true positives, false positives, false negatives and true negatives of the given
        logical formula.
        Args:
            ground_truth: The ground truth which should be used for evaluation.
            logical_formula: The logical formula which should be evaluated.

        Returns:
            A tuple containing the sizes of the true positives, false positives, true negatives and false negatives of
            the given logical formula.
        """
        explanation_set = self._eval_formula(logical_formula)
        true_positives = len(explanation_set & ground_truth)
        false_positives = len(explanation_set - ground_truth)
        false_negatives = len(ground_truth - explanation_set)
        true_negatives = self.data.num_nodes - true_positives - false_positives - false_negatives

        return true_positives, false_positives, true_negatives, false_negatives

    @functools.lru_cache(maxsize=100)
    def _eval_formula(self, logical_formula: OWLClassExpression) -> set[tuple[int, str]]:
        """
        Evaluates the given logical formula based on the given dataset and returns the set of matching nodes.

        Args:
            logical_formula: The logical formula which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        return self.owl_mapping[type(logical_formula)](logical_formula)

    def _eval_complement(self, logical_formula: OWLObjectComplementOf) -> set[tuple[int, str]]:
        """
        Evaluates the given complement based on the given dataset and returns the set of matching nodes.
        which are the complement of the inner set.

        Args:
            logical_formula: The complement which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        inner_set = self._eval_formula(logical_formula.get_operand())
        return self._nodeset - inner_set

    def _eval_union(self, logical_formula: OWLObjectUnionOf) -> set[tuple[int, str]]:
        """
        Evaluates the given union based on the given dataset and returns the set of matching nodes.
        Args:
            logical_formula: The union which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        operands = list(logical_formula.operands())
        result = set()
        for i in operands:
            result = result | self._eval_formula(i)
        return result

    def _eval_intersection(self, logical_formula: OWLObjectIntersectionOf) -> set[tuple[int, str]]:
        """
        Evaluates the given intersection based on the given dataset and returns the set of matching nodes.
        Args:
            logical_formula: The intersection which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        operands = list(logical_formula.operands())
        result = self._eval_formula(operands[0])
        for i in operands[1:]:
            result = result & self._eval_formula(i)
        return result

    def _eval_existential(self, logical_formula: OWLObjectSomeValuesFrom) -> set[tuple[int, str]]:
        """
        Evaluates the given existential based on the given dataset and returns the set of matching nodes.
        Args:
            logical_formula: The existential restriction which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        dest = self._eval_formula(logical_formula.get_filler())
        edge_type = self._eval_property(logical_formula.get_property())
        dest_first_elements = np.array([b[0] for b in dest])
        selection = np.isin(self.data[edge_type]['edge_index'][1].cpu(), dest_first_elements)
        origin = self.data[edge_type]['edge_index'][0][selection].cpu().numpy()
        return set(zip(origin, [edge_type[0], ] * len(origin)))

    def _eval_object_one_of(self, logical_formula: OWLObjectOneOf) -> set[tuple[int, str]]:
        """
        Evaluate an OWL ObjectOneOf logical formula and return a set of tuples
        representing nodes that match the condition.

        Args:
            logical_formula: The OWL ObjectOneOf logical formula to evaluate.

        Returns:
            A set of tuples where each tuple represents a node that matches the condition.
            Each tuple contains two elements: an integer representing the index and a string representing the node type.
        """
        nodes = set()
        individuals = list(logical_formula.individuals())
        for individual in individuals:
            node_type, index = individual.get_iri().get_remainder().split('#')
            nodes.add((int(index), node_type))
        return nodes

    def _eval_universal(self, logical_formula: OWLObjectAllValuesFrom) -> set[tuple[int, str]]:
        """
        Evaluates the given universal based on the given dataset and returns the set of matching nodes.
        Args:
            logical_formula: The universal restriction which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        dest = set(self._eval_formula(logical_formula.get_filler()))
        edge_type = self._eval_property(logical_formula.get_property())
        result = set()

        mapping = dict()

        # Convert edge_index arrays to NumPy arrays for better performance
        edge_index_0 = self.data[edge_type]["edge_index"][0].cpu().numpy()
        edge_index_1 = self.data[edge_type]["edge_index"][1].cpu().numpy()

        for i in range(len(edge_index_0)):
            idx_0 = edge_index_0[i].item()
            idx_1 = edge_index_1[i].item()

            if idx_0 not in mapping:
                mapping[idx_0] = [idx_1]
            else:
                mapping[idx_0].append(idx_1)

        for i, indices in mapping.items():
            check_set = {(idx, edge_type[2]) for idx in indices}
            if check_set.issubset(dest):
                result.add((i, edge_type[0]))

        return result

    def _eval_max_cardinality(self, logical_formula: OWLObjectMaxCardinality) -> set[tuple[int, str]]:
        """
        Evaluates the given max cardinality restriction based on the given dataset and returns the set of matching
        nodes.

        Args:
            logical_formula: The max cardinality restriction which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        dest = set(self._eval_formula(logical_formula.get_filler()))
        edge_type = self._eval_property(logical_formula.get_property())
        cardinality = logical_formula.get_cardinality()
        result = set()

        mapping = dict()

        # Convert edge_index arrays to NumPy arrays for better performance
        edge_index_0 = self.data[edge_type]["edge_index"][0].cpu().numpy()
        edge_index_1 = self.data[edge_type]["edge_index"][1].cpu().numpy()

        for i in range(len(edge_index_0)):
            idx_0 = edge_index_0[i].item()
            idx_1 = edge_index_1[i].item()

            if idx_0 not in mapping:
                mapping[idx_0] = [idx_1]
            else:
                mapping[idx_0].append(idx_1)

        for i, indices in mapping.items():
            check_set = {(idx, edge_type[2]) for idx in indices}
            if len(check_set) <= cardinality and check_set.issubset(dest):
                result.add((i, edge_type[0]))

        return result

    def _eval_min_cardinality(self, logical_formula: OWLObjectMinCardinality) -> set[tuple[int, str]]:
        """
        Evaluates the given min cardinality restriction based on the given dataset and returns the
        set of matching nodes.

        Args:
            logical_formula: The min cardinality restriction which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        dest = set(self._eval_formula(logical_formula.get_filler()))
        edge_type = self._eval_property(logical_formula.get_property())
        cardinality = logical_formula.get_cardinality()
        result = set()

        mapping = dict()

        # Convert edge_index arrays to NumPy arrays for better performance
        edge_index_0 = self.data[edge_type]["edge_index"][0].cpu().numpy()
        edge_index_1 = self.data[edge_type]["edge_index"][1].cpu().numpy()

        for i in range(len(edge_index_0)):
            idx_0 = edge_index_0[i].item()
            idx_1 = edge_index_1[i].item()

            if idx_0 not in mapping:
                mapping[idx_0] = [idx_1]
            else:
                mapping[idx_0].append(idx_1)

        for i, indices in mapping.items():
            check_set = {(idx, edge_type[2]) for idx in indices}
            if len(check_set) >= cardinality and check_set.issubset(dest):
                result.add((i, edge_type[0]))

        return result

    def _eval_class(self, logical_formula: OWLClass) -> set[tuple[int, str]]:
        """
        Evaluates the given class based on the given dataset and returns the set of matching nodes.
        Args:
            logical_formula: The class which should be evaluated.

        Returns:
            A set of nodes which are the result of the evaluation.
        """
        return self._get_nodeset([logical_formula.get_iri().get_remainder(), ])

    def _eval_property_value(self, logical_formula: OWLDataSomeValuesFrom) -> set[tuple[int, str]]:
        """
        Evaluates the given OWLDataSomeValuesFrom logical formula based on the dataset and returns the set of nodes
        that satisfy the specified property value condition.

        Args:
            logical_formula: The OWLDataSomeValuesFrom expression representing a property value condition.

        Returns:
            A set of nodes that satisfy the specified property value condition.
                                 Each tuple contains the node index and node type.
        """
        nodes_matching_condition = set()

        # Extract information from the logical formula
        property_iri = logical_formula.get_property().get_iri().get_remainder()
        facet_restriction = logical_formula.get_filler().get_facet_restrictions()[0]

        # Parse property information
        property_split = property_iri.split('_')
        node_type = property_split[0]
        feature_index = int(property_split[-1]) - 1

        # Extract operator and comparison value from facet restriction
        operator = facet_restriction.get_facet().operator
        comparison_value = facet_restriction.get_facet_value()._v

        # Retrieve nodes and evaluate the condition
        nodes = self.data[node_type]['x'].cpu().numpy()
        for index, node in enumerate(nodes):
            if operator(node[feature_index], comparison_value):
                nodes_matching_condition.add((index, node_type))

        return nodes_matching_condition

    def _eval_property(self, property: OWLObjectProperty) -> tuple[str, str, str]:
        """
        Evaluates the given property based on the given dataset and returns the edge type.
        Args:
            property: The property which should be evaluated.

        Returns:
            The edge type which is the result of the evaluation.
        """
        for i in self.data.edge_types:
            if i[1] == property.get_iri().get_remainder():
                return i

    def _get_nodeset(self, node_types: list[str] = None) -> set[tuple[int, str]]:
        """
        Returns the set of nodes of the given node types.
        Args:
            node_types: The node types for which the nodes should be returned.

        Returns:
            The set of nodes of the given node types.
        """
        if node_types is None or node_types == ['Thing', ]:
            node_types = self.data.node_types
        if node_types == ['Nothing', ]:
            return set()
        result = set()
        for i in node_types:
            if "x" in self.data[i]:
                result = result | set(enumerate([i] * self.data[i]["x"].shape[0]))
            elif "num_nodes" in self.data[i]:
                result = result | set(enumerate([i] * self.data[i]["num_nodes"]))
        return result
