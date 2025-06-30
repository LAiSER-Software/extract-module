import pandas as pd

from laiser.skill_extractor import Skill_Extractor



print('\n\nInitializing the Skill Extractor...')
se = Skill_Extractor(AI_MODEL_ID="gemini", api_key = "AIzaSyAaos4KqOBIXX5L0v0Go0ktn2C4AxTU5Qg",HF_TOKEN="args.HF_TOKEN", use_gpu=True)
print('The Skill Extractor has been initialized successfully!\n')

# Skill extraction from jobs data
print('\n\nLoading a sample dataset of 50 jobs...')

job_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/linkedin_jobs_sample_36rows.csv')

job_sample = job_sample[0:1]
job_sample = job_sample[['description', 'job_id']]
print("Considering", len(job_sample), "rows for processing...")


output = se.extractor(job_sample, 'job_id', text_columns = ['description'])
print('The skills have been extracted from jobs data successfully...\n')

# Save the extracted skills to a CSV file
print(output)
file_name = f'extracted_skills_for_{len(job_sample)}Jobs.csv'
output.to_csv(file_name, index=False)
print('The extracted skills have been saved to the file named:', file_name)
