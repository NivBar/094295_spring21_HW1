import torchvision
import torch.nn as nn
import config
import torch
import matplotlib.pyplot as plt


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc


def plot_measurement_graph(train_measure_list_results, test_measure_list_results, measure_name):
    epoch_list = list(range(len(train_measure_list_results)))
    plt.plot(epoch_list, train_measure_list_results, label='train')
    plt.plot(epoch_list, test_measure_list_results, label='test')
    plt.xlabel('Epochs')
    plt.ylabel(measure_name)
    plt.legend()
    plt.savefig(f'{measure_name}_graph.png')
    plt.show()


def train_model_reg(model, criterion, optimizer, train_loader, test_loader, device, arc_name):
    best_test_iou = (0.0, 0.0)
    train_iou_per_epoch, test_iou_per_epoch, train_loss_per_epoch, test_loss_per_epoch = [], [], [], []
    print("training " + arc_name + " reg task")
    for e in range(1, config.epochs + 1):
        # train
        model.train()
        epoch_iou_train = 0.0
        epoch_loss_train = 0.0
        for i, train_data in enumerate(train_loader):
            images, bboxes, _ = train_data
            images = images.float().to(device)
            bboxes = bboxes.to(device)
            X_batch, y_batch = images, torchvision.ops.box_convert(bboxes, 'xywh', 'xyxy')
            optimizer.zero_grad()

            y_pred = model.forward(X_batch).to(device)
            if i % 100 == 0:
                print(f"epoch: {e}, batch: {i + 1}/{len(train_loader)}")
            loss = criterion(y_pred.float(), y_batch.float())
            loss.backward()
            optimizer.step()

            iou_train = torch.sum(torch.diagonal(torchvision.ops.box_iou(y_pred, y_batch), 0)) / len(bboxes)
            epoch_iou_train += iou_train
            epoch_loss_train += loss.item()
        train_iou_per_epoch.append(epoch_iou_train / len(train_loader))
        train_loss_per_epoch.append(epoch_loss_train / len(train_loader))

        model.eval()
        # test
        epoch_iou_test = 0.0
        epoch_loss_test = 0.0
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                images, bboxes, _ = test_data
                images = images.float().to(device)
                bboxes = bboxes.to(device)
                X_batch, y_batch = images, torchvision.ops.box_convert(bboxes, 'xywh', 'xyxy')

                y_pred = model.forward(X_batch).to(device)
                loss = criterion(y_pred.float(), y_batch.float())

                iou_test = torch.sum(torch.diagonal(torchvision.ops.box_iou(y_pred, y_batch), 0)) / len(bboxes)
                epoch_iou_test += iou_test
                epoch_loss_test += loss.item()

        if epoch_iou_test / len(test_loader) > best_test_iou[0]:
            best_test_iou = (epoch_iou_test / len(test_loader), e)
            torch.save(model.state_dict(), f'best_{arc_name}_reg_iou_{best_test_iou[0]}_ep_{best_test_iou[1]}.pkl')

        test_iou_per_epoch.append(epoch_iou_test / len(test_loader))
        test_loss_per_epoch.append(epoch_loss_test / len(test_loader))
        print(
            f'Epoch {e + 0:03}: | Loss_Train: {train_loss_per_epoch[-1]:.5f} | Train_IOU: {train_iou_per_epoch[-1]:.5f}  | Loss_Test: {test_loss_per_epoch[-1]:.5f} | Test_IOU: {test_iou_per_epoch[-1]:.5f}')

    plot_measurement_graph(train_iou_per_epoch, test_iou_per_epoch, f'IOU_reg_graph_{arc_name}')
    plot_measurement_graph(train_loss_per_epoch, test_loss_per_epoch, f'Loss_reg_graph_{arc_name}')


def train_model_clf(model, criterion, optimizer, train_loader, test_loader, device, arc_name):
    best_test_acc = (0.0, 0.0)
    train_acc_per_epoch, test_acc_per_epoch, train_loss_per_epoch, test_loss_per_epoch = [], [], [], []
    print("training " + arc_name + " clf task")
    for e in range(1, config.epochs + 1):
        # train
        model.train()
        epoch_loss_train = 0.0
        epoch_acc_train = 0.0
        for i, train_data in enumerate(train_loader):
            images, _, labels = train_data
            images = images.float().to(device)
            labels = labels.float().to(device)
            X_batch, y_batch = images, labels
            optimizer.zero_grad()

            y_pred = model.forward(X_batch).to(device)
            Sigmoid = nn.Sigmoid()
            y_pred = Sigmoid(y_pred)

            if i % 50 == 0:
                print(f"epoch: {e}, batch: {i + 1}/{len(train_loader)}")

            loss = criterion(y_pred.squeeze(1), y_batch)
            acc = binary_acc(y_pred.squeeze(1), y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss_train += loss.item()
            epoch_acc_train += acc.item()

        train_acc_per_epoch.append(epoch_acc_train / len(train_loader))
        train_loss_per_epoch.append(epoch_loss_train / len(train_loader))

        model.eval()
        # test
        epoch_acc_test = 0.0
        epoch_loss_test = 0.0
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                images, _, labels = test_data
                images = images.float().to(device)
                labels = labels.float().to(device)
                X_batch, y_batch = images, labels

                y_pred = model.forward(X_batch).to(device)
                Sigmoid = nn.Sigmoid()
                y_pred = Sigmoid(y_pred)
                acc = binary_acc(y_pred.squeeze(1), y_batch)
                loss = criterion(y_pred.squeeze(1), y_batch)

                epoch_acc_test += acc.item()
                epoch_loss_test += loss.item()
        test_acc_per_epoch.append(epoch_acc_test / len(test_loader))
        test_loss_per_epoch.append(epoch_loss_test / len(test_loader))

        if epoch_acc_test / len(test_loader) > best_test_acc[0]:
            best_test_acc = (epoch_acc_test / len(test_loader), e)
            torch.save(model.state_dict(), f'best_{arc_name}_clf_acc_{best_test_acc[0]}_ep_{best_test_acc[1]}.pkl')

        print(
            f'Epoch {e + 0:03}: | Train_Loss: {epoch_loss_train / len(train_loader):.5f} | Train_Acc: {epoch_acc_train / len(train_loader):.3f}% |  Test_Loss: {epoch_loss_test / len(test_loader):.5f} | Test_Acc: {epoch_acc_test / len(test_loader):.3f}%')

    plot_measurement_graph(train_acc_per_epoch, test_acc_per_epoch, f'acc_clf_graph_{arc_name}')
    plot_measurement_graph(train_loss_per_epoch, test_loss_per_epoch, f'Loss_clf_graph_{arc_name}')
