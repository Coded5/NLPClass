import pandas as pd
import sys, os
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', 10) # for google colab
pd.set_option('display.max_rows', 10) # for google colab

def get_test_id(df):
    id_set = set(df["id"].tolist())
    return id_set


class Check_ID():
    def __init__(self, test_file, pred_file):
        self.test_file = test_file
        self.pred_file = pred_file

    def check_files(self):
        if not os.path.exists(self.test_file):
            print(f'NOT FOUND: {self.test_file}'); sys.exit()
        if not os.path.exists(f'{self.pred_file}'):
            print(f'NOT FOUND: {self.pred_file}'); sys.exit()
        try:
            self.test = pd.read_csv(f'{self.test_file}')
        except:
            print(f'CANNOT READ: {self.test_file}'); sys.exit()
        try:
            self.pred = pd.read_csv(f'{self.pred_file}')
        except:
            print(f'CANNOT READ: {self.pred_file}'); sys.exit()
        if not {'id','aspectCategory','polarity'} <= set(self.pred.columns):
            print(f'INCORRECT COLUMN NAME : must have "id", "aspectCategory", "polarity"'); sys.exit()

    def match_all_id(self):
        # added to check if all id are present
        test_id = get_test_id(self.test)
        ID_used_in_prediction = get_test_id(self.pred)
        if len(test_id.difference(ID_used_in_prediction)) > 0:
            print("There are missing IDs:", test_id.difference(ID_used_in_prediction)); sys.exit()
        if len(ID_used_in_prediction.difference(test_id)) > 0:
            print("There are unexpected IDs:", ID_used_in_prediction.difference(test_id)); sys.exit()
        print("All good! Sentences in your predicted file are labeled!")

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 3:
        print('INCORRECT ARGUMENTS!!')
        print('try : $ python check_id.py contest1_test.csv test_pred.csv')
        sys.exit()
    CHECK = Check_ID(args[1], args[2])
    CHECK.check_files()
    CHECK.match_all_id()
