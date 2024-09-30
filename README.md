# Incident classification based on early warnings in a monitoring system for business continuity management

## About the Project
To reduce the impact on businesses during crisis events such as natural disasters, wars and pandemics, organisational resilience is crucial for companies to develop business continuity strategies, such as the Business Continuity Management System (BCMS). Identifying possible disruptive events is essential for a company's success, and it is necessary to invest in risk monitoring technologies and processes, such as screening previous alerts from various sources and critically analysing the information collected. The article highlights the use of Machine Learning techniques, such as the Naive Bayes model and Support Vector Machine, to increase the accuracy of incident identification, making the monitoring process more efficient and accurate, showing that the Support Vector Machine model had a high accuracy metric and F1-score, 98.8026% and 92.8794% respectively, as well as a favourable AUC score.
![image](https://github.com/user-attachments/assets/4a92a9f5-b4a3-4a4d-badd-983a88dee8eb)

## Technical Implementation
The project employs the following models and tools:

- **Natural Language Processing (NLP)**:NLP bridges the gap between computers and human language, enabling machines to understand and process spoken and written words as we do. It was used for transcribe text in number language for computer comprehension.

- **Train-test split**: train test split technique was used for separate the dataset.
![image](https://github.com/user-attachments/assets/c2dee2c8-5ce5-4a66-bfe3-284aa347471d)

- **Data Balancing**: This model is employed for re-ranking queries post-expansion, known for its effectiveness in evaluating query-document relevance.
![image](https://github.com/user-attachments/assets/962fae53-9385-4702-9d9c-24fe0e6a6c04)

- **Support Vector Machine**: This model is employed for classify binaries variables (e.g.: either incident or not-incident)

- **Multinomial Naive Bayes**: This model is employed for classify text in spite of its capability of dealing with large amount of variables.

## Conclusions and Results
Based on the results, it is possible to conclude that the automatic classification of incidents is feasible with the SVM and NB Multinomial machine learning models as they showed high performance in identifying incidents in a data set of previous alerts, with the SVM model being the most efficient, allowing for better identification of incidents when analysing alerts. The F-score metric was considered the most appropriate for evaluating the performance of the models, since it focuses on the efficiency of identifying true positives. Both models had a high F1-score, with SVM standing out with 92.8794%. The ROC curve also indicated high performance for both models, with an AUC of 0.98 for SVM and 0.97 for Multinomial NB. It can therefore be concluded that the use of machine learning models to classify incidents is important and effective for the BCM area and other areas of organisational resilience, since the rapid identification of an incident helps an organisation to prepare for and shield itself from any unplanned event that may occur.
![image](https://github.com/user-attachments/assets/2e96ed3d-fe62-4c28-a67f-58dc92af7420)

