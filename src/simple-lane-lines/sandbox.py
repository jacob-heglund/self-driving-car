from collections import deque

my_list = [[1,2], [3,4], [5,6], [7,8]]
num_history = 3
d = deque(maxlen = 3)

def deque_avg(deque, num_history):
    """Find the average of each component in a list of lists

    Args:
        deque (deque): deque = [[a1, b1], [a2, b2], ...]
        a historical list of the parameters of linear fit
    """
    # f(x) = ax + b
    a_avg = 0
    b_avg = 0
    if len(deque) < num_history:
        num_history = len(deque)

    for i in range(num_history):
        a_avg += deque[i][0]
        b_avg += deque[i][1]

    return a_avg / num_history, b_avg / num_history

for i in range(4):
    d.append(my_list[i])

    a, b = deque_avg(d, num_history)
    print("Linear Average:", a)
    print("Const. Average:", b)