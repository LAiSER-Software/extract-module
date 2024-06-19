"""
Module Description:
-------------------
Class to extract skills from text and align them to existing taxonomy

Ownership:
----------
Project: LAISER
Owner: GW PSCWP

License:
--------
© 2024 Organization Name. All rights reserved.
Licensed under the XYZ License. You may obtain a copy of the License at
http://www.example.com/license


Input Requirements:
-------------------
- Input files or data formats required by the module.

Output/Return Format:
----------------------------
- Description of the output files or data formats produced by the module.


Revision History:
-----------------
Rev No.     Date            Author              Description
[1.0.0]     06/08/2024      Satya Phanindra K.  Initial Version


TODO:
-----
- 1: Import and test the Skill_Extraction class
- 2: Combine OSN and ESCO taxonomies into one taxonomy
- 3: Change the output to a JSON format
"""

import pandas as pd

# Load the CSV file into a Pandas DataFrame
osn_comp_df = pd.read_csv('data/osn_comp.csv')
osn_ind_df = pd.read_csv('data/osn_ind.csv')
osn_pr_df = pd.read_csv('data/osn_pr.csv')

# Create a new column 'ID' to store the modified WGUSID values
osn_comp_df['ID'] = ''
osn_ind_df['ID'] = ''
osn_pr_df['ID'] = ''

# Iterate over each row in the DataFrame
for df in [osn_comp_df, osn_ind_df, osn_pr_df]:
    # Iterate over the 'Keywords' column
    for index, keywords in df['Keywords'].items():
        # Split the keywords by ';'
        parts = keywords.split(';')
        
        # Iterate over the parts
        for part in parts:
            # Check if the part starts with 'WGUSID:'
            if part.strip().startswith('WGUSID:'):
                # Split the part to get the unique number
                unique_number = part.split(':')[1].strip()
                
                # Construct the new ID in the desired format
                new_id = f'OSN.{unique_number}'
                
                # Assign the new ID to the 'ID' column for the current row
                df.at[index, 'ID'] = new_id
                
                # Break out of the inner loop since we found the WGUSID
                break

# Save the updated DataFrame to a new CSV file
osn_comp_df.to_csv('data/osn_comp_prepped.csv', index=False)
osn_ind_df.to_csv('data/osn_ind_prepped.csv', index=False)
osn_pr_df.to_csv('data/osn_pr_prepped.csv', index=False)

from Skill_Extractor import Skill_Extractor

# Create an instance of the class
skill_extractor = Skill_Extractor(taxonomy='LIGHTCAST')

# Extract skills from text
input_text = """SANCORP is seeking FTE Level II Data Scientist to support the office of DoD Chief Digital and Artificial Intelligence Office (CDAO) Chief Technology Officer (CTO). CDAO CTO requires support in multiple functional areas to ensure deliverables associated with the CDAO Architecture Council, CTO Federation, and CTO Future Architecture Activities. The mission of the CDAO CTO is to accelerate the DoD's adoption of data, analytics, and AI to improve decision making across all levels of the department. The following are examples of responsibilities:

    Support development of insider threat strategy in support of protecting CDAO technical offerings; balance short-term wins with long-term investments to progressively mature CDAO’s defenses against insider threats.
    Lead coordination of policy and strategy related to insider threats with industry partners and other DoD components.
    Lead exploration of data sources that are relevant to measuring, identifying, and defending against insider threats.
    Provide technical leadership in developing capabilities to detect insider threats among large user communities, leveraging combination of statistical, classical machine learning, and deep learning methods.

Sancorp Consulting LLC shall, in its discretion, modify or adjust the position to meet Sancorp’s changing needs. This job description is not a contract and may be adjusted as deemed appropriate at Sancorp’s sole discretion.

Sancorp Consulting, LLC, is an SDVOSB and SBA 8(a) company seeking highly motivated and qualified professionals and offer an attractive salary and benefits package that includes: Medical, Dental, life and Disability Insurance; 401K, and holidays to ensure the highest quality of life for our employees. Please visit our website for more information at www.sancorpconsulting.com.

Sancorp Consulting, LLC is an equal opportunity employer. At Sancorp Consulting, LLC we are committed to providing equal employment opportunities (EEO) to all employees and applicants without regard to race color, religion, sex, national origin, age, disability, or any other protected characteristic as defined by applicable law. We strive to create an inclusive and diverse workplace where everyone feels valued, respected, and supported."""
extracted_skills = skill_extractor.extract(input_text)
print("Extracted Skills:", extracted_skills)

# Align skills with the 'OSN' taxonomy
aligned_skills = skill_extractor.get_aligned_skills(extracted_skills)
print("Aligned Skills (OSN):", aligned_skills)

# # Align skills with the 'ESCO' taxonomy
# aligned_skills_esco = skill_extractor.get_aligned_skills(extracted_skills, output_taxonomy='ESCO')
# print("Aligned Skills (ESCO):", aligned_skills_esco)

