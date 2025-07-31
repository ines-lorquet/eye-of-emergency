class DecisionTree:
    def __init__(self, feature_names, target_names):
        self.feature_names = feature_names
        self.target_names = target_names
        self.tree = None

    def _count_classes(self, data):
        counts = {}
        labels = data['label'].values
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def _gini(self, data):
        counts = self._count_classes(data)
        impurity = 1
        total = len(data)
        if total == 0:
            return 0.0

        for lbl in counts:
            prob_of_lbl = counts[lbl] / total
            impurity -= prob_of_lbl ** 2
        return impurity

    class _Question:
        def __init__(self, column, value, feature_names):
            self.column = column
            self.value = value
            self.feature_names = feature_names

        def match(self, example):
            val = example[self.column]
            return val >= self.value

        def __repr__(self):
            condition = ">="
            return f"Est-ce que {self.feature_names[self.column]} {condition} {self.value:.3f}?"

    class _Leaf:
        def __init__(self, data):
            self.predictions = {}
            if len(data) > 0:
                self.predictions = DecisionTree._count_classes(None, data)

    class _DecisionNode:
        def __init__(self, question, true_branch, false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch

    def _split_data(self, data, question):
        true_rows = data[data.apply(lambda row: question.match(row), axis=1)]
        false_rows = data[~data.apply(lambda row: question.match(row), axis=1)]
        return true_rows, false_rows

    def _find_best_split(self, data):
        best_gain = 0
        best_question = None
        current_gini = self._gini(data)
        n_features = len(self.feature_names)

        if len(data) == 0:
            return 0, None

        for col in range(n_features):
            values = data.iloc[:, col].unique()

            for val in values:
                question = self._Question(col, val, self.feature_names)

                true_rows, false_rows = self._split_data(data, question)

                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                p = float(len(true_rows)) / len(data)
                gain = current_gini - (p * self._gini(true_rows) + (1 - p) * self._gini(false_rows))

                if gain >= best_gain:
                    best_gain = gain
                    best_question = question

        return best_gain, best_question

    def _build_tree(self, data):
        gain, question = self._find_best_split(data)

        if gain == 0 or len(data) == 0:
            return self._Leaf(data)

        true_rows, false_rows = self._split_data(data, question)

        true_branch = self._build_tree(true_rows)
        false_branch = self._build_tree(false_rows)

        return self._DecisionNode(question, true_branch, false_branch)

    def fit(self, data):
        self.tree = self._build_tree(data)

    def _classify(self, row, node):
        if isinstance(node, self._Leaf):
            return node.predictions

        if node.question.match(row):
            return self._classify(row, node.true_branch)
        else:
            return self._classify(row, node.false_branch)

    def predict(self, X_data):
        predictions = []
        for index, row in X_data.iterrows():
            prediction_counts = self._classify(row, self.tree)
            if prediction_counts:
                predicted_label = max(prediction_counts, key=prediction_counts.get)
            else:
                predicted_label = None
            predictions.append(predicted_label)
        return predictions

    def print_tree(self, node=None, spacing=""):
        if node is None:
            node = self.tree

        if isinstance(node, self._Leaf):
            total_samples = sum(node.predictions.values())
            if total_samples > 0:
                probabilities = {self.target_names[k]: v/total_samples for k, v in node.predictions.items()}
                print(f"{spacing}Prédit: {probabilities}")
            else:
                print(spacing + "Prédit: {}")
            return

        print(spacing + str(node.question))

        print(f'{spacing}--> Vrai:')
        self.print_tree(node.true_branch, f"{spacing}  ")

        print(f'{spacing}--> Faux:')
        self.print_tree(node.false_branch, f"{spacing}  ")