# Implementation of simple cascaded face alignment by pytorch
We provide pytorch code for simple cascaded face alignment.
This code take some other codes from [D-X-Y/landmark-detection/SAN](https://github.com/D-X-Y/landmark-detection/tree/master/SAN)

## Preparation

### Dependencies
- Python3.7
- PyTorch=1.3
- scipy
- Pillow
- opencv-python
- progress

### Datasets Download
- [300W(part1)](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.001)
- [300W(part2)](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.002)
- [300W(part3)](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.003)
- [300W(part4)](https://ibug.doc.ic.ac.uk/download/annotations/300w.zip.004)
- [ibug](https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip)
- [afw](https://ibug.doc.ic.ac.uk/download/annotations/afw.zip)
- [helen](https://ibug.doc.ic.ac.uk/download/annotations/helen.zip)
- [lfpw](https://ibug.doc.ic.ac.uk/download/annotations/lfpw.zip)
- [bounding_boxex](https://ibug.doc.ic.ac.uk/media/uploads/competitions/bounding_boxes.zip)

Extract all downloaded files into one directory(ex. ./datasets/300W, ./datasets/ibug, ...)

### Makeing File List for Training
```
python make_300W_train_list.py ./datasets 300W_train.txt
```

## Training and Test

### Training
```
python train.py
```

### Test
```
python test.py
```

