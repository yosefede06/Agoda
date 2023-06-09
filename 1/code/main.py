from hackathon_code import task_1, task_2, task_3
import pandas as pd
root = "1/code/"
import joblib

if __name__ == "__main__":
    # ======= QUESTION 1 =======
    task_1.task_1(pd.read_csv(root + "Agoda_Test_1.csv"), pd.read_csv(root + "agoda_cancellation_train.csv"))
    # ======= QUESTION 2 =======
    task_2.task_2(pd.read_csv(root + "Agoda_Test_2.csv"), pd.read_csv(root + "agoda_cancellation_train.csv"))
    # ======= QUESTION 3 =======
    task_3.task_3()
    # ======= QUESTION 4 =======
    # task_4.main_task4()


# def task_2():
#     # ======= QUESTION 2 =======
#     X, y, cancel_y = clean_data_regression()
#     apply_model_regression(X, y, cancel_y)