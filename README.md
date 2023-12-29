# United-States-Census-income-prediction
Build an efficient classification model to predict whether a person's yearly income is above $50k or not

## Introduction:
US Adult Census data which is used for this project has around 41 independent features such as Age, Education, race, occupation, age, native country, race, capital gain, capital loss, education, work class and more. Each row is labelled as -50000 and 50000+.

- Raw data has lot of junk and missing values, so thorough exploration of training data is required

Details about the dataset count is as below,

Training data:
    Number of records: 199523
    Number of Independent Features: 41
Validation data:
    Number of records: 99762
    Number of Independent Features: 41

## Dependencies
- Anaconda3
- Jupyter Notebook or any IDE like VS code
- Python==3.10.11
- pandas==2.1.4
- numpy==1.26.2
- matplotlib==3.8.2
- scikit-learn==1.3.2
- seaborn==0.13.0
- joblib==1.3.2
- flask

## Steps to use the .ipynb file using Anaconda jupyter notebook:
0. Clone the project to the local
1. Open Anaconda prompt
2. Create a new environment
```
conda create -name env_name python==3.10.11 -y

```
3. Activate the environment
```
conda activate env_name

```
4. Install other dependent libraries
```
conda install --file requirements.txt
```
5. Install juperter notebook in the new environment
```
pip install jupyter

```
6. Open Jupyter Notebook
```
jupyter notebook

```
7. Open 01-census_income-prediction.ipynb file from the browser
8. Execute cell by cell 

## API usage in Local:
0. Clone the project to the local
1. Open cmd prompt
2. Make you have python==3.10.11
3. Make sure all other libraries are as per the versions in requirement.txt file
4. Run the below command
```
python app.py

```
5. open another command prompt and go to the project path
6. copy and paste the below command to check the prediction
```
curl -X POST -H "Content-Type: application/json" -d @input.json http://127.0.0.1:5000/predict
```
7. To check the output for various test data replace the values in input.json with valid values from validation dataset
8. We can create a new json by changing the index value of last line of .ipynb file and running it.
   

