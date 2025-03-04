import pandas as pd
from laiser.skill_extractor import Skill_Extractor

print('\n\nInitializing the Skill Extractor...')
se = Skill_Extractor() # runs __init__() method
print('The Skill Extractor has been initialized successfully!\n')


# Skill extraction from jobs data
print('\n\nLoading a sample dataset of 50 jobs...')
nlx_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/nlx_job_data_50rows.csv')
print('The sample jobs dataset has been loaded successfully!\n')


nlx_sample = nlx_sample[['description', 'job_id']]
nlx_sample = nlx_sample[1:3]
print('The sample dataset has been filtered successfully!\n')
print('Head of the sample:\n', nlx_sample.head())

output = se.extractor(nlx_sample, 'job_id', text_columns = ['description'])
print('The skills have been extracted from jobs data successfully...\n')

# save the extracted skills to a csv file
print(output)
file_name = f'extracted_skills_for_{len(nlx_sample)}Jobs.csv'
output.to_csv(f'{file_name}', index=False)
print('The extracted skills have been saved to the file named:', file_name)


# Skill extraction from syllabi data
print('\n\nLoading a sample dataset of 50 syllabi...')
syllabi_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/syllabi-data/preprocessed_50_opensyllabus_syllabi_data.csv')
print('The sample syllabi dataset has been loaded successfully!\n')

syllabi_sample = syllabi_sample[['id', 'description', 'learning_outcomes']]
syllabi_sample = syllabi_sample[1:3]
print('The sample dataset has been filtered successfully!\n')
print('Head of the sample:\n', syllabi_sample.head())

output = se.extractor(syllabi_sample, 'id', text_columns=['description', 'learning_outcomes'], input_type='syllabus')
print('The skills have been extracted from syllabi data successfully...\n')

# save the extracted skills to a csv file
print(output)
file_name = f'extracted_skills_for_{len(syllabi_sample)}Syllabi.csv'
output.to_csv(f'{file_name}', index=False)
print('The extracted skills have been saved to the file named:', file_name)