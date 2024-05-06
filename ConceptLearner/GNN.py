"""Methods and classes which represent an abstract GNN (graph neural network)."""
from abc import ABC, abstractmethod

from torch_geometric.data import HeteroData


class GNN(ABC):
    """ An abstract class representing a GNN which is able to predict if a node is of a certain class."""

    @abstractmethod
    def predict(self, data: HeteroData, node_type: str, node_id: int, label: int) -> bool:
        """Predicts of the given node is of a certain label, based on previous training.

        Args:
            data: The dataset which is used for prediction.
            node_type: The type of the node which should be predicted.
            node_id: The id of the node which should be predicted.
            label: The label for which it should be determined if the node is of this label.

        Return:
            If the node is of the label or not.
        """
        pass

    @abstractmethod
    def predict_all(self, new_data: HeteroData) -> dict:
        """
        Predicts all nodes of the given dataset.

        Args:
            new_data: The dataset which is used for prediction.

        Return:
            A dictionary containing the predictions for each node type.
        """
        pass
