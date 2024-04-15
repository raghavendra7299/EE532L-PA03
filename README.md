# EE532L Deep Learning for Healthcare - Programming Assignment 03

## Report (pa03_a.py)
Performance metrics: Test Accuracy: 0.76623

Figures:![Screenshot 2024-04-15 001932](https://github.com/Electrical-Engineering-IIT-Tirupati/EE532L-PA03/assets/159757647/d7c621ed-c43f-459f-acba-9bdb5c118f89)
![Screenshot 2024-04-15 001952](https://github.com/Electrical-Engineering-IIT-Tirupati/EE532L-PA03/assets/159757647/5be99a46-9730-4fe8-8e43-9338d94fb478)


![fig](assets/logo.png)

Results and Observations: The feedforward neural network achieved a test accuracy of approximately 76.62%. The model showed a decreasing trend in loss over the epochs, indicating that it was learning the underlying patterns in the data. Overall, the model demonstrated moderate performance in classifying individuals as diabetic or non-diabetic based on the given attributes, but further optimization and tuning may be necessary to improve its accuracy and generalization.

## Report (pa03_b.ipynb)
Performance metrics: Test Accuracy: 0.625

Figures:![Screenshot 2024-04-15 084602](https://github.com/Electrical-Engineering-IIT-Tirupati/EE532L-PA03/assets/159757647/81ea4fd8-a695-4371-b9d1-5444cea54888)
![Screenshot 2024-04-15 083347](https://github.com/Electrical-Engineering-IIT-Tirupati/EE532L-PA03/assets/159757647/e9df0954-4603-4419-a202-31890466a17a)

Results and Observations:For both configurations of the neural network, with two hidden layers of 16 neurons each and with three hidden layers of 64, 32, and 16 neurons respectively, the accuracy achieved on the dataset was approximately 62.5%. Despite the deeper architecture of the second configuration, which might suggest a potential for higher performance, both models yielded similar results. This could indicate that the complexity added by the additional hidden layers and neurons did not significantly improve the model's ability to classify pneumonia from normal chest X-ray images in this particular dataset. Further experimentation with different architectures, hyperparameters, and possibly data augmentation techniques may be necessary to improve the model's performance.

## About (pa03_a.py)
The Pima Indians Diabetes Database is a widely used dataset in machine learning, particularly for binary classification tasks related to diabetes prediction. The dataset consists of 768 instances. There are 8 numeric predictive attributes. The attributes are as follows:
 - Pregnancies: Number of times pregnant
 - Glucose: Plasma glucose concentration 2 hours in an oral glucose tolerance test
 - BloodPressure: Diastolic blood pressure (mm Hg)
 - SkinThickness: Triceps skin fold thickness (mm)
 - Insulin: 2-Hour serum insulin (mu U/ml)
 - BMI: Body mass index (weight in kg/(height in m)^2)
 - DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores the likelihood of diabetes based on family history)
 - Age: Age in years

The target variable is a binary variable indicating whether a patient has diabetes or not. It takes the values 0 (no diabetes) or 1 (diabetes). Now your goal is to build a feed forward neural network from scratch, which includes 8 input neurons and 1 hidden layer with 3 neurons, to accurately classify individuals as diabetic or non-diabetic based on the given attributes.
This time you also have to perform validation. You have to plot graphs of all the metrics as done in the first programming assignment. Each graph should have both training as well as validation curve. If you notice overfitting then address it accordingly and write your observations 

## About (pa03_b.ipynb)
The PneumoniaMNIST consists of 5,856 pediatric chest X-Ray images. The task is binary-class classification of pneumonia against normal. The #Training/ Validation/Test is 4,708/524/624. Image size is 1 × 28 × 28.

![fig](assets/pos.png)
![fig](assets/neg.png)

The target variable is a binary variable indicating whether a patient has pneumonia or not. Now your goal is to build a feed forward neural network using tensorflow to accurately classify.
For this we have already provided you with a skeleton code, you just have to fill the snippets and tune the parameters. Similarly you have to plot the training and validation curves for all the metrics. You have to tune the hyperparameters accordingly if you notice overfitting in your plotted graphs.

## Instructions
  - Make sure you have a GitHub account. If you don't have one, create an account at [GitHub](https://github.com/).
  - Please accept this invite (shared in the Google Classroom) for the GitHub Classroom.
  - Once you accept the assignment invitation, you will be redirected to your assignment repository on GitHub.
  - You're supposed to only change the sections where you are allowed to do so in the pa03_a.py script.
  - Then upload or commit and push the changes to your assignment repo.
  - Your assignment will be automatically graded, otherwise, you will receive an email that the auto-grading has failed then make sure the code has no errors.

## References
- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
- Yang, Jiancheng, et al. "Medmnist v2-a large-scale lightweight benchmark for 2d and 3d biomedical image classification." Scientific Data 10.1 (2023): 41.


## License and Acknowledgement
The dataset is from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data?select=diabetes.csv) and [MedMNIST](https://medmnist.com/). Please follow their licenses. Thanks for their awesome work.

