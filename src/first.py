import numpy as np

def Train(train):
    w = np.random.rand(3)
    #print(w)
    #print(train)
    w_train = np.dot(train, w)
    #print(w_train)
    return (w_train >= 1.0).astype(int)


if __name__ == "__main__":
    train = np.random.randint(0, 2, (6, 3))
    print(f"Train result: {Train(train)}")
    test = np.random.randint(0, 2, (1000, 3))
    print(f"Train result: {Train(test)}")