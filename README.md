<div align="center">
<img src="https://github.com/user-attachments/assets/2b7a9900-68db-49ef-9fe2-0da8f4ffa6b0" width="70%"/>
</div>

# Contents
LAiSER stands for  Leveraging ​Artificial ​Intelligence for ​Skill ​Extraction &​ Research​. LAiSER is a tool that helps learners, educators and employers share trusted and mutually intelligible information about skills​.

- [About](#about)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Examples](#examples)

## 1. About
## 2. Requirements
- Python version >= Python 3.12. 
- A GPU with atelast 15GB video memory is essential for running this tool on large datasets.
- If you encounter any `*.dll` file missing errors, make sure you downgrade the pytorch version to `2.2.2`.
```shell
pip install pytorch=2.2.2
```

## 3. Setup and Installation

### i. Download the repository
Before proceeding to  LAiSER, you'd want to follow the steps below to install the required dependencies:
- Clone the repository using 
  ```shell
  git clone https://github.com/Micah-Sanders/LAiSER.git
  ```
  or download the [zip(link)](https://github.com/Micah-Sanders/LAiSER/archive/refs/heads/main.zip) file and extract it.

### ii. Install the dependencies
> [!NOTE]
> If you intend to use the Jupyter Notebook interface, you can skip this step as the dependencies will be installed seperately in the Google Colab environment.

Install the required dependencies using the command below:
  ```shell
    pip install -r requirements.txt
```
**NOTE**: Python 3.9 or later, *preferably 3.12*, is expected to be installed on your system. If you don't have Python installed, you can download it from [here](https://www.python.org/downloads/).


## 4. Usage

As of now LAiSER can be used a command line tool or from the Jupyter notebook(Google Colab). The steps to setup the tool are as follows:

### Google Colab Setup(preferred)
LAiSER's Jupyter notebook is, currently, the fastest way to get started with the tool. You can access the notebook [here](/notebooks/Extract%20Function%20Colab%20Execution.ipynb)

- Once the notebook is imported in google colaboratory, connect to a GPU-accelerated runtime(T4 GPU) and run the cells in the notebook.

### Command Line Setup
To use LAiSER as a command line tool, follow the steps below:

- Navigate to the root directory of the repository and run the command below:
  ```shell
  python main.py
  ```

## 5. Examples


## Funding

## Authors

## Partners
<p align='center'> <b> Made with Passion💖, Data Science📊, and a little magic!🪄 </b></p>
