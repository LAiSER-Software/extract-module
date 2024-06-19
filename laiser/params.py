
"""
Module Description:
-------------------
A python file with global constants

Ownership:
----------
Project: Leveraging Artificial intelligence for Skills Extraction and Research (LAiSER)
Owner:  George Washington University Institute of Public Policy
        Program on Skills, Credentials and Workforce Policy
        Media and Public Affairs Building
        805 21st Street NW
        Washington, DC 20052
        PSCWP@gwu.edu
        https://gwipp.gwu.edu/program-skills-credentials-workforce-policy-pscwp

License:
--------
Copyright 2024 George Washington University Institute of Public Policy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
"""
Revision History
-----------------
Rev No.     Date            Author              Description
[1.0.0]     06/01/2024      Vedant M.           Initial Version
[1.0.1]     06/10/2024      Vedant M.           added paths for input and output


TODO:
-----
- 1:
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(ROOT_DIR, 'input')
OUTPUT_PATH = os.path.join(ROOT_DIR, 'output')

SKILL_DB_PATH = os.path.join(INPUT_PATH, 'combined.csv')

# with open(DATA_PATH+'skill_db_relax_20.json') as json_file:
#     SKILL_TAXONOMY = json.load(json_file)


SIMILARITY_THRESHOLD = 0.05

AI_MODEL_ID = 't:davinci-002:personal::8IIFVUbf'
API_KEY = 'sk-proj-aXyKoTNJA8Cd4E7fP9s0T3BlbkFJKVQSd6folsAavHi3mQWi'

