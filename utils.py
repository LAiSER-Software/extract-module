"""
Module Description:
-------------------
A Class with utility functions

Ownership:
----------
Project: LAISER
Owner: GW PSCWP

License:
--------
© 2024 Organization Name. All rights reserved.
Licensed under the XYZ License. You may obtain a copy of the License at
http://www.example.com/license



Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     06/01/2024      Vedant M.           Initial Version


TODO:
-----
- 1: 
"""

import numpy as np


class Utils:
    """
    A utility class for LAISER.

    ...

    Parameters (optional from __init__ function)
    ----------
    param1 : type
        Short description
    param2 : type
        Short description
        Defaults to ''
        
    Methods
    -------
    get_embedding(input_text)
        Creates vector embeddings for input text based on nlp object

    ....

    """

    def __init__(self, nlp):
        """
        Constructor for Utility class

        Parameters
        ----------
        nlp : spacy nlp model
        """
        # Initialization of objects here
        self.nlp = nlp
        return

    def get_embedding(self, input_text):
        """
        Creates vector embeddings for input text based on nlp object

        Parameters
        ----------
        input_text : text
            Provide text to be vectorized, usually skill, extracted of referenced

        Returns
        -------
        numpy array of vectorized text

        """
        nlp = self.nlp
        doc = nlp(input_text)
        if len(doc) == 0:
            return np.zeros(300)  # Return zeros for empty texts
        return np.mean([word.vector for word in doc], axis=0)

    def cosine_similarity(self, vec1, vec2):
        """
        Calculates cosine similarity between 2 vectors

        Parameters
        ----------
        vec1, vec2 : numpy array of vectorized text

        Returns
        -------
        numeric value
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
