import numpy as np 
import os
import argparse

def split_data(main_dir, train_ratio, test_ratio):

    image_paths= []

    train_paths = []
    val_paths = []
    test_paths = []

    for i, classx in enumerate(os.listdir(main_dir)): #classes
        class_path = os.listdir(os.path.join(main_dir, classx))
        for class_img in class_path:
            image_paths.append([class_img, classx])
        np.random.shuffle(image_paths)
        train_filenames, val_filenames, test_filenames = np.split(np.array(image_paths),
                                                            [int(len(image_paths)*train_ratio), int(len(image_paths)*(train_ratio+test_ratio))])
        print("******", classx, "********")
        print(len(train_filenames), len(val_filenames), len(test_filenames))

        train_paths.extend([img_name for img_name in train_filenames.tolist()]) 
        val_paths.extend([img_name for img_name in val_filenames.tolist()])
        test_paths.extend([img_name for img_name in test_filenames.tolist()])

    np.savetxt('train.csv', train_paths, delimiter=',', fmt ='% s')
    np.savetxt('validation.csv', val_paths, delimiter=',', fmt ='% s')
    np.savetxt('test.csv', test_paths, delimiter=',', fmt ='% s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Project')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--dataset-path', type=str, default=None)

    args = parser.parse_args()
    print(args)

    split_data(args.dataset_path, args.train_ratio, args.test_ratio)