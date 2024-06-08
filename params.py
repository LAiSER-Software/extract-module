
"""
Module Description:
-------------------
A python file with global constants

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

import json

DATA_PATH = './data/'

with open(DATA_PATH+'skill_db_relax_20.json') as json_file:
    SKILL_TAXONOMY = json.load(json_file)


SIMILARITY_THRESHOLD = 0.65

