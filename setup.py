from setuptools import setup, find_packages

setup(
    name='dev-laiser',
    version='0.2.24', 
    author='Satya Phanindra Kumar Kalaga, Prudhvi Chekuri, Bharat Khandelwal', 
    author_email='phanindra.connect@gmail.com',  
    description='LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) is a tool designed to help learners, educators, and employers extract and share trusted information about skills. It uses a fine-tuned language model to extract raw skill keywords from text, then aligns them with a predefined taxonomy. You can find more technical details in the project’s paper.md and an overview in the README.md.', 
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/LAiSER-Software/extract-module',  
    packages=find_packages(),  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'psutil',
        'skillNer',
        'scikit-learn',
        'spacy',
        'transformers',
        'tokenizers',
        'torch',
        'ipython',
        'python-dotenv',
        'vllm',
        'tqdm'
    ],

)