def evaluate(epoch, net):

  net = net.to(DEVICE)

  net.train(False) # Set Network to evaluation mode

  running_loss=0
  correct=0
  total=0

  running_corrects = 0

  for images, labels in val_dataloader:
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    # Forward Pass
    outputs = net(images)

    # Compute loss based on output and ground truth
    loss = criterion(outputs, labels)
    running_loss+=loss.item()

    # Get predictions
    _, preds = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += preds.eq(labels).sum().item()

    # Update Corrects
    running_corrects += torch.sum(preds == labels.data).data.item()

  # Calculate Accuracy
  accuracy = running_corrects / float(len(val_dataset))

  test_loss=running_loss/len(val_dataloader)
  accu=100.*correct/total
 
  eval_losses.append(test_loss)
  eval_accu.append(accu)

  print('Valid Loss: %.3f | Validation Accuracy: %.3f'%(test_loss,accu))

  return accu