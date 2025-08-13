"""
Module Description:
-------------------
Defines custom exception classes for the LAiSER project to handle various error conditions in a structured way.

Ownership:
----------
Project: Leveraging Artificial intelligence for Skills Extraction and Research (LAiSER)
Owner:  George Washington University Insitute of Public Policy
    Program on Skills, Credentials and Workforce Policy
    Media and Public Affairs Building
    805 21st Street NW
    Washington, DC 20052
    PSCWP@gwu.edu
    https://gwipp.gwu.edu/program-skills-credentials-workforce-policy-pscwp

License:
--------
Copyright 2025 George Washington University Insitute of Public Policy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
(the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, 
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


Input Requirements:
-------------------
- No specific input files required. This module is imported and used by other modules to raise custom exceptions.

Output/Return Format:
----------------------------
- Provides custom exception classes for error handling in the LAiSER project.

"""

"""
Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     08/13/2025      Satya Phanindra K.            Initial Version 

TODO:
-----

"""

class LAiSERError(Exception):
    """Base exception class for LAiSER project"""
    pass

class ModelLoadError(LAiSERError):
    """Raised when model loading fails"""
    pass

class VLLMNotAvailableError(LAiSERError):
    """Raised when vLLM is required but not available"""
    pass

class SkillExtractionError(LAiSERError):
    """Raised when skill extraction fails"""
    pass

class FAISSIndexError(LAiSERError):
    """Raised when FAISS index operations fail"""
    pass

class InvalidInputError(LAiSERError):
    """Raised when input validation fails"""
    pass
