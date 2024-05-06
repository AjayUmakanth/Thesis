"""Methods and classes that inherit the parent explainer class"""
from ConceptLearner import GNN
from ConceptLearner import ConvertToOWL
from ConceptLearner.Utils import _find_classes_with_y_labels
from torch_geometric.data import HeteroData
from ontolearn.owlapy.model import OWLClassExpression
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.concept_learner import EvoLearner
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.owlapy.model import OWLNamedIndividual, IRI
from ontolearn.metrics import Accuracy, F1
from ontolearn.abstracts import AbstractScorer
from typing import Optional
import os
from ontolearn.owlapy.model import OWLNamedIndividual, IRI, OWLObjectIntersectionOf, \
    OWLClassExpression, OWLObjectUnionOf, OWLObjectComplementOf, OWLObjectOneOf, \
    OWLObjectMaxCardinality, OWLObjectMinCardinality, OWLObjectAllValuesFrom, \
    OWLObjectSomeValuesFrom, OWLClass, OWLNothing
import random
import re

class DiscriminativeExplainer():
    """ An abstract class which represent an explainer. An explainer should be able to use a label to generate a
    model-level explanation based on a given GNN and the related dataset.

    Attributes:
        gnn: The GNN the class is able to explain.
        data: The dataset the GNN is based on.
    """
    def __replace_with_nominal(self, ce: OWLClassExpression): #, top_nodes_dict: dict[list[int], str]):
        if isinstance(ce, OWLObjectIntersectionOf):
            ops = list()
            for op in ce.operands():
                mutated = self.__replace_with_nominal(op)
                ops.append(mutated)
            return OWLObjectIntersectionOf(ops)
        if isinstance(ce, OWLObjectUnionOf):
            ops = list()
            for op in ce.operands():
                mutated = self.__replace_with_nominal(op)
                ops.append(mutated)
            return OWLObjectUnionOf(ops)
        if isinstance(ce, OWLObjectComplementOf):
            return OWLObjectComplementOf(self.__replace_with_nominal(ce.get_operand()))
        if isinstance(ce, OWLClass):
            className = ce.get_iri().get_remainder()
            isNominal = re.findall(r'_\d+$', className)
            if not isNominal:
                return ce
            else:
                splits = className.rsplit("_", 1)
                individual = OWLNamedIndividual(IRI("/", splits[0]+'#'+splits[1]))
                return OWLObjectOneOf(individual)
        if isinstance(ce, OWLObjectSomeValuesFrom):
            ce2 = self.__replace_with_nominal(ce.get_filler())
            return OWLObjectSomeValuesFrom(ce.get_property(), ce2)
        if isinstance(ce, OWLObjectAllValuesFrom):
            ce2 = self.__replace_with_nominal(ce.get_filler())
            return OWLObjectAllValuesFrom(ce.get_property(), ce2)
        if isinstance(ce, OWLObjectMaxCardinality):
            ce2 = self.__replace_with_nominal(ce.get_filler())
            return OWLObjectMaxCardinality(ce.get_cardinality(), ce.get_property(), ce2)
        if isinstance(ce, OWLObjectMinCardinality):
            ce2 = self.__replace_with_nominal(ce.get_filler())
            return OWLObjectMinCardinality(ce.get_cardinality(), ce.get_property(), ce2)
        return ce

    def __init__(self, gnn: GNN, data: HeteroData, namespace = "http://example.org/", owl_graph_path = "./owlGraphs/example.owl", generate_new_owl_file: bool = False, create_nominals: bool = False,  add_edge_counts: bool = False) -> None:
        """Initializes the explainer based on the given GNN and the Dataset. After the initialization the object should
        be able to produce explanations of single labels.

        Args:
            gnn: The GNN the explainer should be able to explain.
            data: The data the GNN was trained with and which will be used for explanation.
            namespace: The namespace to which we have to assign the nodes of a graph to.
            create_nominals: Create seperate classes for each individual to support nominals
        """
        self.gnn = gnn
        self.data = data
        self.namespace = namespace
        self.owl_graph_path = owl_graph_path
        self.create_nominals = create_nominals
        self.classNames = _find_classes_with_y_labels(self.data, first_only=False)
        if generate_new_owl_file and os.path.isfile(self.owl_graph_path):
            os.remove(self.owl_graph_path)
        if not os.path.isfile(self.owl_graph_path):
            self.owlGraph = ConvertToOWL(data=self.data, namespace=self.namespace, owlGraphPath=self.owl_graph_path, create_nominals=create_nominals, add_edge_counts=add_edge_counts)
            self.owlGraph.buildGraph()
        self.knowledge_base = KnowledgeBase(path=self.owl_graph_path)

    def explain(self, 
            label: int,
            n: Optional[int] = 5,
            use_data_properties: Optional[bool] = True,
            debug: Optional[bool] = False,
            max_runtime: Optional[int] = 60,
            num_generations: Optional[int] = 600,
            quality_func: Optional[AbstractScorer] = None) -> OWLClassExpression:
        """Explains based on the GNN a given label. The explanation is in the form of a Class Expression.

        Args:
            label: The label which should be explained.
            n: the maximum number of hypotheses that will be produced
            debug: use the y label in the node of the heterodata object
            use_data_properties: Use the data properties in the explainer
            max_runtime (int): max_runtime: Limit to stop the algorithm after n seconds
            num_generations (int): Number of generation for the evolutionary algorithm

        Returns:
            A class expression which is an explanation of the given label based on the GNN and the dataset.
        """

        if quality_func is None:
            quality_func = F1()
        self.model = EvoLearner(knowledge_base=self.knowledge_base, use_data_properties=use_data_properties, max_runtime=max_runtime, num_generations=num_generations, quality_func=quality_func)
        positive_examples = []
        negative_examples = []
        predictions = {}
        if (not debug):
            predictions = self.gnn.predict_all(new_data=self.data)
        for node_type in self.data.node_types:
            if (debug):
                if node_type in self.classNames:
                    yLabels = self.data[node_type].y
                    for idx, yLabel in enumerate(yLabels):
                        node = self.namespace + node_type + "#" + str(idx+1)
                        isPositive = yLabel.item() == label
                        if (isPositive):
                            positive_examples.append(node)
                        else:
                            negative_examples.append(node)
                else:
                    noOfNodes = self.data[node_type].x.size()[0]
                    for idx in range(noOfNodes):
                        node = self.namespace + node_type + "#" + str(idx+1)
                        negative_examples.append(node)
            else:
                if node_type in predictions:
                    nodeTypePredictions = predictions[node_type]
                    for idx, nodePrediction in enumerate(nodeTypePredictions):
                        node = self.namespace + node_type + "#" + str(idx+1)
                        isPositive = nodePrediction.item() == label
                        if (isPositive):
                            positive_examples.append(node)
                        else:
                            negative_examples.append(node)
                else:
                    noOfNodes = self.data[node_type].x.size()[0]
                    for idx in range(noOfNodes):
                        node = self.namespace + node_type + "#" + str(idx+1)
                        negative_examples.append(node)
        
        if len(positive_examples) == 0:
            return [OWLNothing]

        typed_pos = set(map(OWLNamedIndividual, map(IRI.create, set(positive_examples))))
        typed_neg = set(map(OWLNamedIndividual, map(IRI.create, set(negative_examples))))
        lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
        
        # cross check if the quality stated is within the threshold of the actual quality
        while True:
            self.model.fit(lp)
            hypotheses = list(self.model.best_hypotheses(n=n))
            accepted_hypotheses = []
            for hypothesis in hypotheses:
                evaluated_concept = self.model.kb.evaluate_concept(hypothesis.concept, self.model.quality_func, self.model._learning_problem)
                thresh = 0.05
                accepted_hypotheses.append(hypothesis.quality*(1+thresh) > evaluated_concept.q or hypothesis.quality*(1-thresh) < evaluated_concept.q)

            hypotheses = [element for element, keep in zip(hypotheses, accepted_hypotheses) if keep]

            if len(hypotheses) > 0:
                break
        return hypotheses, self.model

        [print(_) for _ in hypotheses]
        if self.create_nominals:
            hypotheses = [self.__replace_with_nominal(hypothesis.concept) for hypothesis in hypotheses]
        else:
            hypotheses = [hypothesis.concept for hypothesis in hypotheses]

        return hypotheses
