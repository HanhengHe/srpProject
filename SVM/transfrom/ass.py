from multiprocessing import  Pool


def f(xx, ii):
    return "new"


if __name__ == "__main__":
    pool = Pool(processes=4)
    x = [None, None, None, None]
    for i in range(4):
        x[i] = pool.apply_async(f, (x, i)).get()

    print(x)
