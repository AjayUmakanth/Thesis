from rdflib import XSD, Graph, Literal, RDF, RDFS, Namespace, OWL
from torch_geometric.data import HeteroData
from rdflib.term import URIRef

class ConvertToOWL():
    def __init__(self, data: HeteroData, namespace: str, owlGraphPath: str, create_data_properties: bool = True, create_data_properties_as_object: bool = False, add_edge_counts: bool = False, create_nominals: bool = False) -> None:
        """Converts the heterodata into an OWL format

        Args:
            data (HeteroData): The data the GNN was trained with and which will be used for explanation.
            namespace (str): The namespace on which the graph will be built.
            owlGraphPath (str): Where the graph will be stored.
            create_data_properties (bool): Flag to create data properties in the OWL graph.
            add_edge_counts (bool): Flag to add edge counts to the OWL graph.
            create_nominals (bool): Flag to create nominal classes for node features in the OWL graph.

        """

        self.dataset = data
        self.namespace = namespace
        self.owlGraphPath = owlGraphPath
        self.create_nominals = create_nominals
        self.create_data_properties = create_data_properties
        self.add_edge_counts = add_edge_counts
        self.create_data_properties_as_object = create_data_properties_as_object
        self.graph = Graph()

    #Builds the OWL graph using the provided heterogenous data.
    def buildGraph(self):
        self._createClasses()
        self._buildObjectProperties()
        if (self.create_data_properties):
                self._buildDataProperties()
        self._buildNodes()
        self._buildEdges()
        if (self.add_edge_counts):
            self._buildEdgeCounts()
        self.graph.serialize(self.owlGraphPath, format="xml")
        
    #Creates OWL classes based on node types in the dataset.
    def _createClasses(self):
        classNamespace = Namespace(self.namespace)
        for node in self.dataset.node_types:
            self.graph.add((classNamespace[node], RDF.type, OWL.Class))
            if self.create_nominals:
                n = 0
                if "x" in self.dataset[node]:
                    n = self.dataset[node].x.size(dim=1)
                if "num_nodes" in self.dataset[node]: 
                    n = self.dataset[node].num_nodes
                for i in range(n):
                    self.graph.add((classNamespace[f'{node}_{i+1}'], RDF.type, OWL.Class))
                    self.graph.add((classNamespace[f'{node}_{i+1}'], RDFS.subClassOf, classNamespace[node]))
    
    #Builds OWL datatype properties (attributes) for each node type in the heterodata.
    def _buildDataProperties(self):
        classNamespace = Namespace(self.namespace)
        for node in self.dataset.node_types:
            if "x" in self.dataset[node]:
                n = self.dataset[node].x.size(dim=1)
                for i in range(n):
                    propertyObjectPropertyName = f'{node}_property_{i+1}'
                    if "xKeys" in self.dataset[node]:
                        propertyObjectPropertyName =  self.dataset[node].xKeys[i]   
                    if self.create_data_properties_as_object:
                        propertyObjectPropertyName = "has_" + propertyObjectPropertyName
                    propertyObjectProperty = classNamespace[propertyObjectPropertyName]
                    xsdRange = XSD.boolean if self.create_data_properties_as_object else XSD.double
                    self.graph.add((propertyObjectProperty, RDF.type, OWL.DatatypeProperty))
                    self.graph.add((propertyObjectProperty, RDFS.domain, classNamespace[node]))
                    self.graph.add((propertyObjectProperty, RDFS.range, xsdRange))
    
    #Builds OWL object properties (relationships) based on each edge types in the dataset.
    def _buildObjectProperties(self):
        classNamespace = Namespace(self.namespace)
        for edgeType in self.dataset.edge_types:
            s, p, o = edgeType
            self.graph.add((classNamespace[p], RDF.type, OWL.ObjectProperty))
            self.graph.add((classNamespace[p], RDFS.domain, classNamespace[s]))
            self.graph.add((classNamespace[p], RDFS.range, classNamespace[o]))

    #Builds individual nodes in the OWL graph.
    def _buildNodes(self):
        nodes = self.dataset.node_types
        classNamespace = Namespace(self.namespace)
        rdf = RDF
        for node in nodes:
            if "x" in self.dataset[node]:
                tensorValues = self.dataset[node].x
                for row_idx, person in enumerate(tensorValues):
                    newNode = classNamespace[f'{node}#{row_idx+1}']
                    if self.create_nominals:
                        self.graph.add((newNode, rdf.type, classNamespace[f'{node}_{row_idx+1}']))
                    else:
                        self.graph.add((newNode, rdf.type, classNamespace[node]))
                    self.graph.add((newNode, rdf.type, OWL.NamedIndividual))
                    if self.create_data_properties:
                        for col_idx, property in enumerate(person):
                            val = property.item()
                            propertyObjectPropertyName = f'{node}_property_{col_idx+1}'
                            if "xKeys" in self.dataset[node]:
                                propertyObjectPropertyName =  self.dataset[node].xKeys[col_idx]
                            if self.create_data_properties_as_object:
                                propertyObjectPropertyName = "has_" + propertyObjectPropertyName
                            propertyObjectProperty = classNamespace[propertyObjectPropertyName]
                            if propertyObjectPropertyName == "17th_century" or propertyObjectPropertyName == "th_century":
                                print(propertyObjectPropertyName)
                                print(propertyObjectProperty)
                            #self.graph.add((newNode, classNamespace[f'has_{node}_property_{col_idx+1}'], classNamespace[f'{node}_property_{col_idx+1}']))
                            if self.create_data_properties_as_object:
                                val = True if val != 0 else False
                            self.graph.add((newNode, propertyObjectProperty, Literal(val)))
            if "num_nodes" in self.dataset[node]: 
                num_nodes = self.dataset[node].num_nodes
                for idx in range(num_nodes):
                    newNode = classNamespace[f'{node}#{idx+1}']
                    self.graph.add((newNode, rdf.type, classNamespace[node]))
                    self.graph.add((newNode, rdf.type, OWL.NamedIndividual))

    #Builds edges between nodes in the OWL graph.
    def _buildEdges(self):
        edgeTypes = self.dataset.edge_types
        classNamespace = Namespace(self.namespace)
        for edgeType in edgeTypes:
            s, p, o = edgeType
            edges = self.dataset[edgeType].edge_index
            colSize = edges.size()[1]
            for col in range(colSize):
                s_idx = edges[0][col].item()
                o_idx = edges[1][col].item()
                self.graph.add((classNamespace[f'{s}#{s_idx+1}'], classNamespace[p], classNamespace[f'{o}#{o_idx+1}']))

    #Builds OWL datatype properties for edge counts.
    def _buildEdgeCounts(self):
        classNamespace = Namespace(self.namespace)
        nodeCounts = {}

        for edge_type in self.dataset.edge_types:
            s, _, o = edge_type
            edge_index = self.dataset[edge_type].edge_index
            nodes, counts = edge_index[0].unique(return_counts=True)
            for node, count in zip(nodes, counts):
                if s not in nodeCounts:
                    nodeCounts[s] = {}
                if node.item()+1 not in nodeCounts[s]:
                    nodeCounts[s][node.item()+1] = { "outgoing": 0, "incoming": 0 }
                nodeCounts[s][node.item()+1]["outgoing"] += count.item()
            
            nodes, counts = edge_index[1].unique(return_counts=True) 
            for node, count in zip(nodes, counts):
                if o not in nodeCounts:
                    nodeCounts[o] = {}
                if node.item()+1 not in nodeCounts[o]:
                    nodeCounts[o][node.item()+1] = { "outgoing": 0, "incoming": 0 }
                nodeCounts[o][node.item()+1]["incoming"] += count.item()

        for nodeType in nodeCounts:
            self.graph.add((classNamespace[f'{nodeType}_incoming'], RDF.type, OWL.DatatypeProperty))
            self.graph.add((classNamespace[f'{nodeType}_incoming'], RDFS.domain, classNamespace[node]))
            self.graph.add((classNamespace[f'{nodeType}_incoming'], RDFS.range, XSD.double))
            self.graph.add((classNamespace[f'{nodeType}_outgoing'], RDF.type, OWL.DatatypeProperty))
            self.graph.add((classNamespace[f'{nodeType}_outgoing'], RDFS.domain, classNamespace[node]))
            self.graph.add((classNamespace[f'{nodeType}_outgoing'], RDFS.range, XSD.double))
            for node in nodeCounts[nodeType]:
                newNode = classNamespace[f'{nodeType}#{node}']
                self.graph.add((newNode, classNamespace[f'{nodeType}_incoming'], Literal(nodeCounts[nodeType][node]["incoming"])))
                self.graph.add((newNode, classNamespace[f'{nodeType}_outgoing'], Literal(nodeCounts[nodeType][node]["outgoing"])))


if __name__ == "__main__":
    from torch_geometric.datasets import DBLP
    dataset = DBLP(root="./datasets/dblp")
    owlGraph = ConvertToOWL(data=dataset.data, namespace="http://example.org/", owlGraphPath = "./owlGraphs/example1.owl")
    owlGraph.buildGraph()