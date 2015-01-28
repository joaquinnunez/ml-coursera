""" Question3 of Machine learning course second quizz """


def gradient_descent(t0, t1, alpha, training_set):
    m = len(training_set)
    h = lambda t0, t1, x: t0 + (t1 * x)

    # theta0
    new_theta0 = 0
    for x, y in training_set:
        new_theta0 += h(t0, t1, x) - y
    new_theta0 = alpha / m * new_theta0
    new_theta0 = t0 - new_theta0

    new_theta1 = 0
    for x, y in training_set:
        new_theta1 += (h(t0, t1, x) - y) * x
    new_theta1 = alpha / m * new_theta1
    new_theta1 = t1 - new_theta1

    return new_theta0, new_theta1

if __name__ == "__main__":
    training_set = [(1, 1), (2, 2), (3, 3)]
    t0 = 0
    t1 = 0.5
    alpha = 0.1
    print(gradient_descent(t0, t1, alpha, training_set))
