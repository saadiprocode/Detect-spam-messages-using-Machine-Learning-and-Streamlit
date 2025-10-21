# src/evaluate.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report




def print_report(model, X_test, y_test):
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))




def plot_confusion(model, X_test, y_test):
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()