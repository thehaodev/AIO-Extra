import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def frequency(a, x):
    count = 0
    for i in a:
        if i == x:
            count += 1
    return count


# Input: Series Output: Float
def gini_impurity(value_counts):
    n = value_counts.sum()
    p_sum = 0
    for key in value_counts.keys():
        p_sum = p_sum + (value_counts[key] / n) * (value_counts[key] / n)
    gini = 1 - p_sum
    return gini


# Show Gini ımpurity measures
def show_gini_impurity_measures():
    plt.figure()
    x = np.linspace(0.01, 1)
    y = 1 - (x * x) - (1 - x) * (1 - x)
    plt.plot(x, y)
    plt.title('Gini Impurity')
    plt.xlabel("Fraction of Class k ($p_k$)")
    plt.ylabel("Impurity Measure")
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.show()


# Input: class_name, data_frame, attribute_name
def gini_for_each_condition(attribute_name, data_frame, class_name):
    attribute_values = data_frame[attribute_name].value_counts()
    gini_a = 0
    for key in attribute_values.keys():
        df_k = data_frame[class_name][data_frame[attribute_name] == key].value_counts()
        n_k = attribute_values[key]
        n = data_frame.shape[0]
        gini_a = gini_a + ((n_k / n) * gini_impurity(df_k))
    return gini_a


# In this sample we will find
def run_simple_test_1():
    # Defining a simple dataset
    data = {
        'love_math': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
        'love_art': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no'],
        'love_english': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
        'love_ai': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'no']}
    df = pd.DataFrame(data, columns=data.keys())

    # GINI for Love AI
    class_name = "love_ai"
    class_value_counts = df[class_name].value_counts()
    print(f'Number of samples in each class is:\n{class_value_counts}')

    gini_class = gini_impurity(class_value_counts)
    print(f'\nGini Impurity of the class is {gini_class:.3f}')

    # Calculating  gini impurity for each condition
    gini_love_math = gini_for_each_condition("love_math", data_frame=df, class_name=class_name)
    gini_love_art = gini_for_each_condition("love_art", data_frame=df, class_name=class_name)
    gini_love_english = gini_for_each_condition("love_english", data_frame=df, class_name=class_name)
    print("gini_love_math", round(gini_love_math, 3))
    print("love_art", round(gini_love_art, 3))
    print("gini_love_english", round(gini_love_english, 3))


# In this sample we will find min and continue step
def run_simple_test_2():
    # Defining a simple dataset
    attribute_names = ['age', 'income', 'student', 'credit_rate']
    class_name = 'default'
    data1 = {
        'age': ['youth', 'youth', 'middle_age', 'senior', 'senior', 'senior', 'middle_age', 'youth', 'youth', 'senior',
                'youth', 'middle_age', 'middle_age', 'senior'],
        'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium',
                   'high', 'medium'],
        'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
        'credit_rate': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair',
                        'excellent', 'excellent', 'fair', 'excellent'],
        'default': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    }
    df1 = pd.DataFrame(data1, columns=data1.keys())

    # Calculating  gini impurity for the attiributes
    class_value_counts = df1[class_name].value_counts()
    print(f'Number of samples in each class is:\n{class_value_counts}')

    gini_class = gini_impurity(class_value_counts)
    print(f'\nGini Impurity of the class is {gini_class:.3f}')
    gini_attiribute = {}
    for key in attribute_names:
        gini_attiribute[key] = gini_for_each_condition(key, data_frame=df1, class_name=class_name)
        print(f'Gini for {key} is {gini_attiribute[key]:.3f}')

    # Compute Gini gain values to find the best split
    # An attribute has maximum Gini gain is selected for splitting.

    min_value = min(gini_attiribute.values())
    print('The minimum value of Gini Impurity : {0:.3} '.format(min_value))
    print('The maximum value of Gini Gain     : {0:.3} '.format(1 - min_value))

    selected_attribute = min(gini_attiribute.keys())
    print('The selected attiribute is: ', selected_attribute)


run_simple_test_2()
