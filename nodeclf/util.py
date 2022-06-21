

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def plot_confusion_matrix(labels, pred_labels):

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    # cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
    cm = metrics.ConfusionMatrixDisplay(cm)

    cm.plot(values_format='d', cmap='Blues', ax=ax)
