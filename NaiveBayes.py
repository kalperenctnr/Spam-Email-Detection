import csv
import numpy as np
from numpy import errstate, isneginf, array, isnan

TRAIN_SIZE = 4085
TEST_SIZE = 1086
# %%
with open(r'x_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    x_test = np.asarray([row for row in csv_reader], dtype=np.float32)

with open(r'y_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    y_test = np.asarray([row for row in csv_reader], dtype=np.float64)

with open(r'x_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    x_train = np.asarray([row for row in csv_reader], dtype=np.float64)

with open(r'y_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    y_train = np.asarray([row for row in csv_reader], dtype=np.float32)
# %%

(unique, counts) = np.unique(y_train, return_counts=True)
frequencies = np.asarray((unique, counts / sum(counts))).T
print(f"Ham Mail Percentage: {100 * frequencies[0][1]:.2f}%")
print(f"Spam Mail Percentage: {100 * frequencies[1][1]:.2f}%")
# %%

spam_indices = y_train == 1
T_spam_j = np.sum(x_train[spam_indices[:, 0]], axis=0)
T_spam = np.sum(x_train[spam_indices[:, 0]])

normal_indices = y_train == 0
T_normal_j = np.sum(x_train[normal_indices[:, 0]], axis=0)
T_normal = np.sum(x_train[normal_indices[:, 0]])
# %%

N_normal = counts[0]
N = sum(counts)
pi_normal = N_normal / N
pi_spam = 1 - pi_normal

theta_spam = T_spam_j / T_spam
theta_normal = T_normal_j / T_normal

# %%


with errstate(divide='ignore'):
    log_theta_spam = np.log(theta_spam)
    log_theta_normal = np.log(theta_normal)

log_theta_spam[isneginf(log_theta_spam)] = np.NINF
log_theta_normal[isneginf(log_theta_normal)] = np.NINF

log_pi_normal = np.log(pi_normal)
log_pi_spam = np.log(pi_spam)
# %%

# If there is an occurence of Nan
# it should be the result of 0*log(0) = 0
with errstate(invalid='ignore'):
    spam_sum = x_test * log_theta_spam
    normal_sum = x_test * log_theta_normal

spam_sum[isnan(spam_sum)] = 0
normal_sum[isnan(normal_sum)] = 0

final_spam_sum = np.sum(spam_sum, axis=1)
final_normal_sum = np.sum(normal_sum, axis=1)

spam = log_pi_spam + final_spam_sum
normal = log_pi_normal + final_normal_sum

results = spam > normal

predictions = results.astype(int)

correct = np.sum(predictions == y_test[:, 0])
print(f"Accuracy of Test Set: {100 * correct / 1086:.3f}%")
print(f"Number of wrong predictions: {TEST_SIZE - correct}")
# %%
import matplotlib.pyplot as plt
import seaborn as sn


def confusion_matrix(predicted, actual):
    c_m = np.zeros((2, 2))
    for i in range(len(actual)):
        c_m[actual[i], predicted[i]] += 1

    return c_m


c_m = confusion_matrix(predictions, y_test.astype(int))
classes = ["Normal", "Spam"]
sn.heatmap(c_m.astype(int), annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()