
# Ames Housing Data and Kaggle Challenge

TLDR: Go to Conclusion

I embarked on a project to create and refine a regression model that accurately predicts the sale price of a house based on a myriad of features. My work revolved around the exceptionally detailed Ames Housing Dataset, which boasts over 70 unique feature columns, providing a comprehensive exploration of residential home attributes.

This challenge was two-fold. I was not only developing an effective predictive model, but I also participated in a Kaggle competition, giving me a chance to put my skills to the test in a real-world setting.

This project served as a journey into the world of regression analysis, a key tool in the data science toolbox. Whether predicting housing prices or solving other problems, the skills I honed throughout this project will prove invaluable in my data science career. Let's dive into my journey with the Ames Housing Dataset!

Problem Statement:

ABC real estate startup in Iowa aims to revolutionize the property investment landscape by leveraging advanced data analytics and predictive modeling. As a tech-forward company, we strive to provide accurate and actionable insights to our clients, enabling them to make informed investment decisions in the highly competitive real estate market.

Our goal is to develop a robust predictive model that can accurately estimate property values in different areas of Iowa. By analyzing various features and factors that influence property prices, we aim to provide investors with reliable predictions and identify lucrative investment opportunities.



## Data Cleaning

The train_data and test_data is cleaned using the function DataCleaner(df)

#### Function: DataCleaner(df)

This function takes in a DataFrame df and returns the cleaned DataFrame. The specific operations performed by this function include:

  - Filling missing values: The function replaces missing values in specific columns with predefined values.

  - Dropping rows with missing values: Some columns are vital enough that any rows missing data in these columns are dropped.

  - Dropping 'Alley' column: This column is deemed unnecessary and is thus removed from the DataFrame.

  - Converting categorical variables: Columns containing categorical data are converted into dummy variables for easier processing during modeling.

  - Predicting and filling missing data: For the 'Lot Frontage' column, the function uses either a linear regression model or a KNeighborsClassifier model to predict and fill missing data based on the 'Lot Area' column.
    
## Key Visualisations

![alt text](correlation.png "Correlation Heatmap")

The correlation heatmap allowed me a quick glance into the relationships between different numerical variables in our dataset. Particularly, it was useful for identifying the features that have strong correlations with our target variable, the Sale Price. This guided me in selecting which features to include in my regression model.

![alt text](Dist_sale_price.png "Sale Price Distribution")

The Sale Price distribution plot offered a view of the range and concentration of our target variable - Sale Price. I observed that the Sale Price was heavily right-skewed, meaning that most houses were sold at lower prices, with a few at exceptionally high prices.

To improve my model's performance, I performed a log transformation on the Sale Price before modeling. This transformation helped to normalize the data and reduce the skewness, leading to a better performing model.

These visualizations served as an integral part of my exploratory data analysis, providing insights into the dataset's underlying structure and guiding my decisions during the modeling process.


## Standardizing the Dataset

Many machine learning algorithms can produce drastically different results based on the scale of features. To circumvent this, I have standardized my numerical features so they are all on the same scale. Standardization of a dataset is a common requirement for many machine learning estimators: they might behave poorly if the individual features do not more or less resemble standard normally distributed data (i.e., Gaussian with 0 mean and unit variance).

I've excluded the 'SalePrice' and 'Id' columns from this process. 'SalePrice' is the target variable, and I want to predict its original values. 'Id' is simply an identifier for each house and does not contain any meaningful information for my model.

## Handling Outliers

In any dataset, outliers can significantly impact the accuracy and reliability of a regression model. These extreme values can skew the model's understanding of the data, leading to less accurate predictions. This is particularly true for the Ames Housing dataset, where property prices can vary wildly based on a multitude of factors.

For this project, I've taken an aggressive approach towards handling outliers, ensuring my model can generalize well to most of the data.

##### Outliers in 1st Floor Square Footage

I identified two major outliers in terms of 1st Floor square footage. These properties have significantly larger 1st floor square footage than the rest of the properties in the dataset. I've removed these outliers to prevent them from skewing my model.

##### Outliers in Year Built

In the dataset, I also discovered two outliers related to the year the houses were built. Specifically, houses built before 1900 were sold for over $300,000 - a price range significantly higher than their contemporaries. I've removed these outliers to ensure my model isn't biased by these extreme cases.

##### Outliers in Overall Quality

Overall quality is a critical factor impacting the price of a property, and it had the highest correlation with Sale Price in the dataset. I've found one major outlier in this category, which I've removed to enhance the reliability of my model.

Removing outliers is a crucial step in pre-processing data for regression modeling. It allows the model to learn from the majority of the data, leading to more accurate and reliable predictions.

## Creating New Features

In order to improve my model's performance, I've added several new features to the dataset. These features are derived from existing variables and aim to provide additional insight into the factors that affect a house's sale price.

Here's a quick overview of the new features and their descriptions:


| Feature | Type | Dataset | Description |
| --- | --- | --- | --- |
| Total_Bathrooms | Float | Train/Test | The total number of bathrooms in the house. This is the sum of full and half bathrooms, including those in the basement. |
| Total_Sq_Ft | Float | Train/Test | The total square footage of the house. This is the sum of the square footage of the first floor, second floor, and basement. |
| Overall_Qual_x_Cond | Integer | Train/Test | A combined measure of the overall quality and condition of the house. This is the product of the 'Overall Qual' and 'Overall Cond' features. |
| Year_Since_Remod | Integer | Train/Test | The number of years since the house was last remodeled. If the house has never been remodeled, this represents the age of the house. |
| If_Remod | Binary | Train/Test | A binary variable indicating whether or not the house has been remodeled. '1' represents a house that has been remodeled, and '0' represents a house that has not. |
| New_Houses | Binary | Train/Test | A binary variable indicating whether the house was sold in the same year it was built. '1' represents a house that was sold the same year it was built, and '0' represents all other houses. |
| House_Age | Integer | Train/Test | The age of the house at the time of sale. |
| Perc_lot_frontage | Float | Train/Test | The percentage of the lot that is frontage. This can give some insight into the layout and potential curb appeal of the property. |
| Perc_rooms_is_bedroom | Float | Train/Test | The ratio of bedrooms to total rooms above grade. This can indicate the layout and functionality of the house's living space. |

The addition of these new features should increase the predictive power of my model. Particularly, the Total_Sq_Ft became the highest correlated numerical feature in the dataset. This shows the importance of feature engineering in improving a model's performance.

## Log Transformation of the Target Variable

The target variable in my dataset, SalePrice, demonstrated significant skewness. Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. When a distribution skews to the right, like my SalePrice, it becomes problematic because models predicated on normally distributed errors struggle to effectively capture the structure and nuances of the data.

Therefore, I've performed a log transformation on SalePrice. The transformation results in a more symmetric, normally distributed target variable, which is more compatible with my model's underlying assumptions. This should ultimately lead to more accurate and generalizable predictions.

The np.log1p() function applies a log(1 + x) transformation to all elements of the column, effectively decreasing the skewness. After this transformation, a skewness closer to zero should be reported, confirming the efficacy of the transformation. Be sure to apply the same transformation to any test data prior to predictions, and reverse it (using np.expm1()) for interpretability of results.

## Converting Numerical Variables to Strings

There are some columns in my dataset that, while they contain numbers, are actually better thought of as categories. For example, a year or an ID number is technically numerical, but it doesn't make sense to perform mathematical operations on them. Adding two years together or taking the average of two ID numbers would not give me meaningful information. Hence, these are essentially categorical variables that are represented with numbers.

The columns 'Id', 'PID', 'MS SubClass', 'Mas Vnr Area', and 'Yr Sold' were converted. Here is why:

'Id': This is just an identifier for each house and does not hold meaningful information for my model.
'PID': This is another identifier for each house.
'MS SubClass': This variable actually contains categories for the type of dwelling involved in the sale. It is better treated as a categorical variable.
'Mas Vnr Area': This variable represents masonry veneer area in square feet. However, it is represented as a categorical variable in the dataset (either the house has a masonry veneer area or it does not).
'Yr Sold': The year the house was sold is also more of a categorical variable, as the year sold can harm our model especially during years of financial unrest.



## The Modeling Process

After completing all the data preprocessing and feature engineering steps, I moved onto the phase of training a model. A critical step in this process involved splitting the data into training and testing sets. This method allows me to assess the model's performance on unseen data, which is essential to evaluate its ability to generalize beyond the data it was trained on.

For the model training, I selected all the present numeric columns as the predictor variables (X_train), and the Sale Price served as the target variable (y_train).

Considering the nature of the data and the problem at hand (predicting a continuous target variable), I decided to use a linear regression model. However, to enhance the model's performance and reduce the risk of overfitting, I utilized a regularized version of linear regression known as Ridge Regression. This model penalizes overly complex models, thus ensuring the model remains more generalized.

Moreover, I integrated cross-validation into the model training process. Cross-validation provides a more robust means to estimate the model's performance. It accomplishes this by partitioning the training set into several subsets and training/testing the model multiple times, each time using a different subset reserved as the test set.

I chose the best model, a Ridge Regression, based on the mean squared error (MSE), a common metric for regression problems that calculates the average squared differences between the predicted and actual values.


## Conclusion

This project involved me employing various data analysis techniques, such as exploratory data analysis (EDA), visualization, feature engineering, and model fitting, to predict house prices using the Ames Housing Dataset.

My key takeaways from the project include:

The EDA process was pivotal in identifying key outliers that could have negatively affected the performance of the linear regression model. This stage also helped me identify high correlation numerical and categorical features, as well as irrelevant feature data, enabling me to reduce the model's variance.

Feature engineering played a significant role in enhancing the model's performance. I created new features that better encapsulated the information in the data. These included 'Total_Bathrooms', 'Total_Sq_Ft', 'Overall_Qual_x_Cond', among others.

I tested different linear regression models, including LassoCV and RidgeCV, which are both regularized versions of linear regression. RidgeCV was determined to provide the best Mean Squared Error (MSE) and R-score values.

I validated the trained model on a test subset of the training data as well as the provided test data. This step confirmed that the model was not underfitting or overfitting the data. With an R-score between 0.90 and 0.94, it was evident that the model is well-prepared to handle new data.

At the time of submission, the trained model achieved an RMSE score of 19832.08715 on Kaggle, ranking 1st in the General Assembly cohort. This accomplishment underlines the effective application of various techniques used in this project and the efficiency of the RidgeCV model in predicting house prices based on the Ames Housing dataset.