import pandas as pd
from laiser.skill_extractor import Skill_Extractor

print('\n\nLoading a sample dataset of 50 jobs...')
nlx_sample = pd.read_csv('https://raw.githubusercontent.com/phanindra-max/LAiSER-datasets/master/nlx_tx_sample_data_gwu.csv')
print('The sample dataset has been loaded successfully!\n')


nlx_sample = nlx_sample[['description', 'job_id']]
nlx_sample = nlx_sample[1:3]
print('The sample dataset has been filtered successfully!\n')
print('Head of the sample:\n', nlx_sample.head())

print('\n\nInitializing the Skill Extractor...')
se = Skill_Extractor() # runs __init__() method
print('The Skill Extractor has been initialized successfully!\n')

output = se.extractor(nlx_sample, 'job_id', 'description')
print('The skills have been extracted successfully...\n')

# save the extracted skills to a csv file
print(output)
output.to_csv('./output/test_extracted_skills_for_2Jobs.csv', index=False)
print('The extracted skills have been saved to the file named: extracted_skills_for_50Jobs.csv')