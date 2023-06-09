import task_1, task_2, task_3, task_4
import pandas as pd
if __name__ == "__main__":
    # ======= QUESTION 1 =======
    task_1.task_1(pd.read_csv("Agoda_Test_1.csv"))
    # ======= QUESTION 2 =======
    task_1.task_2(pd.read_csv("Agoda_Test_2.csv"))
    # ======= QUESTION 3 =======
    # task_1.task_3()
    # ======= QUESTION 4 =======
    # task_1.task_1(pd.read_csv("Agoda_Test_4.csv"))


# def task_2():
#     # ======= QUESTION 2 =======
#     X, y, cancel_y = clean_data_regression()
#     apply_model_regression(X, y, cancel_y)