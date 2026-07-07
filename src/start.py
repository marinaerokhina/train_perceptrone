import numpy as np
import matplotlib.pyplot as mpl
import perceptone as perc

if __name__ == "__main__":
    sp = perc.SimplePerceptone()
    answers = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    train_m = [[1, 1, 1],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 1]]
    #print(f"Train result: {sp.predict(train_m)}\n\tanswers: {answers}"
    #      f"\n\t loss: {sp.train(train_m, answers, 0.1)}")

    p_loss = sp.train(train_m, answers, 0.1)
    mpl.plot(p_loss)
    mpl.xlabel('Эпоха')
    mpl.ylabel('Ошибка (MSE)')
    mpl.grid()
    mpl.show()

    train_new = np.random.randint(0, 2, (1000, 3))
    predictions = sp.predict(train_new)
    final_decisions = (predictions > 0.5).astype(int)
    yes_count = np.sum(final_decisions)
    no_count = len(final_decisions) - yes_count
    print(f"Из 1000 новых людей:\n\tИдут (1): {yes_count}\n\tОстаются (0): {no_count}")
    true_answers = (train_new[:, 0] == 1) & ((train_new[:, 1] == 1) | (train_new[:, 2] == 1))
    true_answers = true_answers.astype(int)
    accuracy = np.mean(final_decisions == true_answers) * 100
    print(f"Точность модели: {accuracy:.2f}%")