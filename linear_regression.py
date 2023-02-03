import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def fake_data():
    stu_id = []
    for i in range(1,1001):
        stu_id.append(i)

    eng_diag = np.random.normal(50,15,1000)

    eng_LEAP = np.random.normal(700, 20, 1000)

    return stu_id, eng_diag, eng_LEAP

def linear_regression():
    data = fake_data()
    stu_id = data[0]
    eng_diag = data[1]
    eng_LEAP = data[2]
    
    x = eng_diag
    y = eng_LEAP

    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()
    
    with open('lin_reg_summary.txt', 'w') as summary_txt:
        summary_txt.write(model.summary().as_text())

    with open('lin_reg_summary.csv', 'w') as summary_csv:
        summary_csv.write(model.summary().as_csv())
    

def scatter_plot():
    data = fake_data()
    stu_id = data[0]
    eng_diag = data[1]
    eng_LEAP = data[2]    
    
    plt.scatter(eng_LEAP, eng_diag)
    plt.ylabel('Eng Diagnostic Score')
    plt.xlabel('Eng LEAP Score')
    plt.title('Linear Regression Analysis of English Diagnostic Scores \nand English LEAP Results')
    plt.savefig('example_scatter_plot.png')
    plt.show()

def main():
    linear_regression()
    scatter_plot()

if __name__ == "__main__":
    main()
    