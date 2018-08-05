import os

img_rows, img_cols, img_size = 224, 224, 224
channel = 3
batch_size = 256
epochs = 10000
patience = 50
num_train_samples = 14883151
num_valid_samples = 2102270
embedding_size = 128
vocab_size = 17628
max_token_length = 40
num_image_features = 2048
hidden_size = 512

train_folder = 'data/ai_challenger_keypoint_train_20170909'
valid_folder = 'data/ai_challenger_keypoint_validation_20170911'
test_a_folder = 'data/keypoint_test_a_images_20180103'
test_b_folder = 'data/keypoint_test_b_images_20180103'
train_image_folder = os.path.join(train_folder, 'keypoint_train_images_20170911')
valid_image_folder = os.path.join(valid_folder, 'keypoint_validation_images_20170911')
test_a_image_folder = os.path.join(test_a_folder, 'keypoint_test_a_images_20180103')
test_b_image_folder = os.path.join(test_b_folder, 'keypoint_test_b_images_20180103')
train_annotations_filename = 'keypoint_train_annotations_20170911.json'
valid_annotations_filename = 'keypoint_validation_annotations_20170911.json'
test_a_annotations_filename = 'keypoint_test_a_annotations_20180103.json'
test_b_annotations_filename = 'keypoint_test_b_annotations_20180103.json'


start_word = '<start>'
stop_word = '<end>'
unknown_word = '<UNK>'

best_model = 'model.03-1.3640.hdf5'
beam_size = 20
