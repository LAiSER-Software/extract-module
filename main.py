import pandas as pd
import argparse
from laiser.skill_extractor import Skill_Extractor

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Skill Extractor on jobs and syllabi data.')
parser.add_argument('--HF_TOKEN', type=str, default=None, help='Hugging Face token for authentication')
parser.add_argument('--AI_MODEL_ID', type=str, default=None, help='Model name for Skill Extractor')
args = parser.parse_args()

print('\n\nInitializing the Skill Extractor...')
se = Skill_Extractor(AI_MODEL_ID=args.AI_MODEL_ID, HF_TOKEN=args.HF_TOKEN)
print('The Skill Extractor has been initialized successfully!\n')

# Skill extraction from jobs data
print('\n\nLoading a sample dataset of 50 jobs...')
nlx_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/nlx_job_data_50rows.csv')
print('The sample jobs dataset has been loaded successfully!\n')

nlx_sample = nlx_sample[['description', 'job_id']]
nlx_sample = nlx_sample[1:3]
print('The sample dataset has been filtered successfully!\n')
print('Head of the sample:\n', nlx_sample.head())

output = se.extractor(nlx_sample, 'job_id', text_columns=['description'])
print('The skills have been extracted from jobs data successfully...\n')

# Save the extracted skills to a CSV file
print(output)
file_name = f'extracted_skills_for_{len(nlx_sample)}Jobs.csv'
output.to_csv(file_name, index=False)
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

# Save the extracted skills to a CSV file
print(output)
file_name = f'extracted_skills_for_{len(syllabi_sample)}Syllabi.csv'
output.to_csv(file_name, index=False)
print('The extracted skills have been saved to the file named:', file_name)