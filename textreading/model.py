from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self) -> None:
        self.rf = RandomForestClassifier(
            n_estimators= 500,
            random_state= 42
        )

    def train(self, x, y):
        self.rf.fit(x, y)

    def test(self, xtest):
        return self.predict(xtest)

    def evaluate(self, y, ytest):
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y, ytest)
        print(f'Score: {str(round(accuracy, 3))}/1')

    def predict(self, x):
        self.rf.predict(x)