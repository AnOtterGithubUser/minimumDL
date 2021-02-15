from loss import CrossEntropyLoss
from optim import Adam
from sklearn.metrics import accuracy_score


def train(seq, dataloader, epochs=10):
    criterion = CrossEntropyLoss(seq)
    optimizer = Adam(seq)
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        n_batch = 0
        for batch, labels in dataloader:
            n_batch += 1
            outputs = seq(batch)
            loss = criterion(outputs, labels)
            accuracy = accuracy_score(outputs.argmax(axis=1), labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss
            epoch_accuracy += accuracy

        print("Epoch {}/{}   -    loss: {:%.5f}   accuracy: {:%.5f}".format(epoch+1,
                                                                            epochs,
                                                                            epoch_loss / n_batch,
                                                                            epoch_accuracy / n_batch))

    print("Finished training !")
