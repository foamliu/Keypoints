import os

image_h, image_w, image_size = 368, 368, 368
channel = 3
batch_size = 256
epochs = 10000
patience = 50
num_train_samples = 210000
num_valid_samples = 30000

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
