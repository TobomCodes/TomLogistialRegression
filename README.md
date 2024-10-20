# TomLogistialRegression
Overview of the Project
Dataset:
I utilized the publicly available wine quality dataset, which contains various chemical properties of wine along with their quality scores, ranging from 0 to 10.

Data Preprocessing:
I began by normalizing the input features to ensure a better fit to the regression model.
Model Development: I implemented the logistic regression algorithm according to the lecture files (see references), defining key functions for calculating predictions, the binary cross-entropy loss, and the derivatives for updating weights and biases.

Training:
I trained the model over multiple epochs, I struggled with fine tuning. No matter which learning rate i used, the final weights were not perfect and the loss was never converging 0. Next time I have to normalize the dataset even more.

Evaluation:
After training, I evaluated the model on a test dataset (The same as the training set). I interpreted the predicted probabilities, classifying them into binary outcomes (1 for good quality and 0 for bad quality) based on a threshold of 0.5. I then calculated the accuracy of the model to determine how often the predictions matched the actual quality ratings. When I trained the weights for 300 epochs, my model could predict the quality of the wine in about 75% of the tests.

Results: The model achieved an accuracy of 75%, demonstrating its capability to predict wine quality based on the chemical attributes. There is a lot to improve on this project, however this would be beyond my scope for this work.

Conclusion:
This project not only enhanced my understanding of logistic regression and its practical applications in machine learning but also my understanding of the language "python".
I look forward to your feedback and any suggestions for future improvements or directions for this project.

Thank you for your giving me this chance of extra work.

Best regards,
Tom Feuersaenger
Student_id :104240190
Machine learning VGU
20/10/24
