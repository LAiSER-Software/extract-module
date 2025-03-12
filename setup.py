from setuptools import setup, find_packages

setup(
    name='laiser',
    version='0.2.0', 
    author='Satya Phanindra Kumar Kalaga, Bharat Khandelwal, Prudhvi Chekuri', 
    author_email='phanindra.connect@gmail.com',  
    description='LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) is a tool designed to help learners, educators, and employers extract and share trusted information about skills. It uses a fine-tuned language model to extract raw skill keywords from text, then aligns them with a predefined taxonomy. You can find more technical details in the project’s paper.md and an overview in the README.md.', 
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/LAiSER-Software/extract-module',  
    packages=find_packages(),  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD-3-Clause License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  
    install_requires=[
        # List your package dependencies here
        # numpy,
        # pandas,
        # psutil,
        # scikit_learn,
        # skillNer,
        # spacy,
        # transformers,
        # accelerate,
        # bitsandbytes,
        # datasets,
        # huggingface_hub,
        # peft,
        # torch,
        # trl,
        # skillNer,
        # ipython,
        # python-dotenv
    ],
)