import numpy as np
import torch
import transforms
from Dataset_300W import Dataset_300W
from CFA import CFA
import cv2
from progress.bar import Bar
from utils.log_utils import AverageMeter, make_inference_image
import options


opt = options.Options(None)
args = opt.opt


def make_train_loader():
    mean_fill   = tuple( [int(x*255) for x in [0.5, 0.5, 0.5] ] )
    normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])
    train_transform = [transforms.PreCrop(args.pre_crop_expand)]
    train_transform += [transforms.TrainScale2WH((args.crop_width, args.crop_height))]
    train_transform += [transforms.AugScale(1.0, args.scale_min, args.scale_max)]
    if args.rotate_max:
        train_transform += [transforms.AugRotate(args.rotate_max)]
    train_transform+= [transforms.AugCrop(args.crop_width, args.crop_height, args.crop_perturb_max, mean_fill)]
    train_transform+= [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose( train_transform )

    train_data = Dataset_300W(args.train_list, args.sigma, train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    return train_loader


def train(train_loader, model, criterion, optimizer, epoch):
    
    # log
    loss_log = AverageMeter()
    bar = Bar('Training', max=len(train_loader))

    model.train()
    for i, (inputs, target, mask) in enumerate(train_loader):
        
        # cuda
        inputs = inputs.cuda()
        target = target.cuda()
        mask = mask.cuda()

        # inference
        outputs = model(inputs)

        # calculate loss
        target = torch.masked_select(target, mask)
        loss = 0
        for output in outputs:
            output = torch.masked_select(output, mask)
            loss += criterion(output, target) / inputs.shape[0]
        loss_log.update(loss.item(), inputs.size(0))

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # show progress
        bar.suffix = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.6f}'.format(
            batch=i + 1,
            size=len(train_loader),
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss_log.avg)
        bar.next()

    bar.finish()

    # save inference image
    cv2.imwrite('{0:06d}.jpg'.format(epoch), make_inference_image(inputs, outputs[-1], mask))


def main():

    # train_loader
    train_loader = make_train_loader()

    # model
    model = CFA(output_channel_num=69)
    model.cuda()

    # criterion and optimizer
    criterion = torch.nn.MSELoss(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    # training
    for epoch in range(args.start_epoch, args.epochs):
        
        # training each epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # save checkpoint file
        state = {'state_dict': model.state_dict()}
        state = model.state_dict()
        filename = 'checkpoints/{0:04d}.pth.tar'.format(epoch)
        torch.save(state, filename)


if __name__ == '__main__':
    main()