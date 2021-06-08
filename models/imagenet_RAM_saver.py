# http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287

from argparse import ArgumentParser
import os
from torchvision import transforms, datasets
import torch as t
import math


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', help="Input directory with source images")
    parser.add_argument('-o', '--out_dir', default="ImageNetRAM/val", help="Output directory for pickle files")
    parser.add_argument('-r', '--resolution', type=int, default=224, help="resolution for dataset")
    args = parser.parse_args()

    return args.in_dir, args.out_dir, args.resolution


def process_folder(in_dir, out_dir, res):

	# OFA ImageNet transformations for validation set
    myTransform = transforms.Compose([
        transforms.Resize(int(math.ceil(res/0.875))),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406],
           std=[
                0.229,
                0.224,
                0.225]
        ),
    ])

    myLoader = t.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(in_dir),
            transform=myTransform
        ),
        batch_size=2500,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )

    # Initial matrices to append to
    baseTensor = t.zeros([1, 3, res, res])
    baseLabels = t.zeros([1]).type(t.LongTensor)

    # Iterate through the data, append
    for i, (images, labels) in enumerate(myLoader):
        print("Batch", i)
        baseTensor = t.cat([baseTensor, images], axis=0)
        baseLabels = t.cat([baseLabels, labels], axis=0)

    # Clip off those initial zero values
    data_val = baseTensor[1:, :, :, :]
    labels_list = baseLabels[1:]

    # Print data dimensions
    print(data_val.shape)
    print(labels_list.shape)

    # Put in dictionary
    d_val = {
        'data': data_val,
        'labels': labels_list
    }

    # Save to disk
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Saving to disk")
    t.save(d_val, os.path.join(out_dir, 'val_data_%d' % res))


if __name__ == '__main__':
    in_dir, out_dir, res = parse_arguments()

    print("Start program ...")
    process_folder(in_dir=in_dir, out_dir=out_dir, res=res)
    print("Finished.")
