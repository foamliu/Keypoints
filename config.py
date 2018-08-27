import os

image_h, image_w, image_size = 368, 368, 368
channel = 3
scale = 8
batch_size = 16
epochs = 10000
patience = 50
weight_decay = 5e-4
base_lr = 4e-5
momentum = 0.9
stages = 6

num_train_samples = 210000
num_valid_samples = 30000
num_joints = 14
num_joints_and_bkg = 15

# 0/右肩	    1/右肘	2/右腕	3/左肩	4/左肘
# 5/左腕	    6/右髋	7/右膝	8/右踝	9/左髋
# 10/左膝	11/左踝	12/头顶	13/脖子

idx_in_raw = [12, 13, 0, 1, 2, 3, 4, 5,
              6, 7, 8, 9, 10, 11]

idx_in_raw_str = [
    'Head', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
    'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle']

joint_pairs = list(zip(
        [12, 13, 0, 1, 13, 3, 4, 0, 6, 7, 3, 9, 10, 13, 13],
        [13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 6, 9]))

num_connections = 15

train_folder = 'data/ai_challenger_keypoint_train_20170909'
valid_folder = 'data/ai_challenger_keypoint_validation_20170911'
test_a_folder = 'data/ai_challenger_keypoint_test_a_20180103'
test_b_folder = 'data/ai_challenger_keypoint_test_b_20180103'
train_image_folder = os.path.join(train_folder, 'keypoint_train_images_20170902')
valid_image_folder = os.path.join(valid_folder, 'keypoint_validation_images_20170911')
test_a_image_folder = os.path.join(test_a_folder, 'keypoint_test_a_images_20180103')
test_b_image_folder = os.path.join(test_b_folder, 'keypoint_test_b_images_20180103')
train_annotations_filename = os.path.join(train_folder, 'keypoint_train_annotations_20170909.json')
valid_annotations_filename = os.path.join(valid_folder, 'keypoint_validation_annotations_20170911.json')
test_a_annotations_filename = os.path.join(test_a_folder, 'keypoint_test_a_annotations_20180103.json')
test_b_annotations_filename = os.path.join(test_b_folder, 'keypoint_test_b_annotations_20180103.json')
