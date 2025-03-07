import numpy as np

def softmax(x,c=7):
    x_array=(np.array(x)-c).astype(np.float64)
    exp = np.exp(x_array)
    return exp/np.sum(exp)

if __name__ == "__main__":
    print(softmax([1,1]))