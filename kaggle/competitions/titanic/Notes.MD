# Titanic competition

## The challenge

The competition is simple: we want you to use the Titanic passenger data (name, age, price of ticket, etc) to try to predict who will survive and who will die.

## Dataset Description

| Variable  | Definition | Key |
| --------  | ---------- | --- |
| survival  | Survival              | 0 = No, 1 = Yes |
| pclass    | Ticket class          | 1 = 1st, 2 = 2nd, 3 = 3rd |
| sex       | Sex                   | |
| Age       | Age in years          | |
| sibsp     | # of siblings / spouses aboard the Titanic | |
| parch     | # of parents / children aboard the Titanic | |
| ticket    | Ticket number         | |
| fare      | Passenger fare        | |
| cabin     | Cabin number          | |
| embarked  | Port of Embarkation   | C = Cherbourg, Q = Queenstown, S = Southampton |

Notes:

**pclass**: A proxy for socio-economic status (SES) 1st = Upper, 2nd = Middle, 3rd = Lower
**sibsp**: Sibling = brother, sister, stepbrother, stepsister; Spouse = husband, wife (mistresses and fiancés were ignored)
**parch**: Parent = mother, father; Child = daughter, son, stepdaughter, stepson; Some children travelled only with a nanny, therefore parch=0 for them.

### Files

#### 1) train.csv

train.csv contains the details of a subset of the passengers on board (891 passengers, to be exact -- where each passenger gets a different row in the table). 
To investigate this data, click on the name of the file on the left of the screen. Once you've done this, you can view all of the data in the window.

#### 2) test.csv

Using the patterns you find in train.csv, you have to predict whether the other 418 passengers on board (in test.csv) survived.

Click on test.csv (on the left of the screen) to examine its contents. Note that test.csv does not have a "Survived" column - this information is hidden from you,
and how well you do at predicting these hidden values will determine how highly you score in the competition!

#### 3) gender_submission.csv

The gender_submission.csv file is provided as an example that shows how you should structure your predictions. It predicts that all female passengers survived, 
and all male passengers died. Your hypotheses regarding survival will probably be different, which will lead to a different submission file. But, just like this file, 
your submission should have:

a "PassengerId" column containing the IDs of each passenger from test.csv.
a "Survived" column (that you will create!) with a "1" for the rows where you think the passenger survived, and a "0" where you predict that the passenger died.