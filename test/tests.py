import unittest
import main as m


class TestTokenize(unittest.TestCase):

    def test_empty_or_no_file(self):
        with self.assertRaises(FileNotFoundError):
            m.tokenize_code('zarif.xyz')

    def test_simple_method(self):
        self.assertEquals(m.tokenize_code('test/examples/simple.py').get('funcs'),
                          ["""def testsum(x, y):
    return x + y"""])

    def test_complex_method(self):
        self.assertEquals(m.tokenize_code('test/examples/complex.py').get('funcs'),
                          ["""def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1"""])

    def test_final(self):
        tokens = m.tokenize_code('test/examples/perceptron.py')
        self.assertEquals(tokens.get('funcs')[0],
                          """def train_perceptron(X_train, y_train, X_test, y_test, eta=0.1, n_epochs=10):
    w = np.zeros(X_train.shape[1])
    for epoch in range(n_epochs):
        for i in range(X_train.shape[0]):
            xi = np.insert(X_train[i], 0, 1)
            yi = y_train[i]
            update = eta * (yi - predict(xi, w))
            w += update * xi
    correct_predictions = 0
    for i in range(X_test.shape[0]):
        xi_test = np.insert(X_test[i], 0, 1)
        yi_test = y_test[i]
        prediction = predict(xi_test, w)
        if prediction == yi_test:
            correct_predictions += 1
    accuracy = correct_predictions / X_test.shape[0]
    return (w, accuracy)""")
        self.assertEquals(tokens.get('funcs')[1], """def predict(xi, w):
    activation = np.dot(xi, w)
    return 1 if activation >= 0 else -1""")
        self.assertEquals(tokens.get('imports')[0], """import numpy as np""")

    if __name__ == '__main__':
        unittest.main()


class TestGenerateComments(unittest.TestCase):

    def test_complex(self):
        tokens = m.tokenize_code('test/examples/complex.py')
        print(m.generate_comments(tokens)[0].content)

    def test_final(self):
        tokens = m.tokenize_code('test/examples/perceptron.py')
        comments = m.generate_comments(tokens)
        print(comments[0].content)
        print(comments[1].content)


class TestWriteComments(unittest.TestCase):

    def test_complex(self):
        tokens = m.tokenize_code('test/examples/complex.py')
        comments = m.generate_comments(tokens)
        m.write_comments(comments, 'test/examples/complex_output.py')

    def test_final(self):
        tokens = m.tokenize_code('test/examples/perceptron.py')
        comments = m.generate_comments(tokens)
        m.write_comments(comments, 'test/examples/perceptron_output.py')
